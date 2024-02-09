from sqlalchemy import create_engine
from datetime import date
import pandas as pd


def generate_queries(tdate: date, version: int, db_schema: str):
    """
    Generates queries for data loading.
    :param tdate: target date
    :param version: target version
    :param db_schema: database schema
    :return: dict with queries
    """

    queries = {
        'src_data': {
            'node_prices': f"""
                            select
                                hour,
                                node,
                                price,
                                unsmoothed_price,
                                u
                            from
                                {db_schema}.src_bus
                            where
                                version = {version}
                                and date = '{tdate.isoformat()}'         
                        """,
            'line_flows': f"""
                            select
                                hour,
                                node_from,
                                node_to,
                                pnum_actual,
                                flow
                            from
                                {db_schema}.src_line
                            where
                                version = {version}
                                and date = '{tdate.isoformat()}'      
                        """,
            'parallel_flows': f"""
                                select
                                    hour,
                                    node_from,
                                    node_to,
                                    pnum,
                                    flow,
                                    flow_to
                                from
                                    {db_schema}.src_parallel
                                where
                                    version = {version}
                                    and date = '{tdate.isoformat()}'      
                            """,
            'rge_pmin_pmax': f"""
                                select
                                    hour,
                                    rge,
                                    pmin_tech,
                                    pmin_heat,
                                    pmin,
                                    pmax,
                                    p
                                from
                                    {db_schema}.src_rge
                                where
                                    version = {version}
                                    and date = '{tdate.isoformat()}'      
                            """,
            'consumer_volumes': f"""
                                    select
                                        hour,
                                        gtp_code,
                                        p,
                                        pmax,
                                        loss
                                    from
                                        {db_schema}.src_consumer
                                    where
                                        version = {version}
                                        and date = '{tdate.isoformat()}'      
                                """,
            'sec_pmin_pmax': f"""
                                select
                                    hour,
                                    sec_num,
                                    coalesce(pmin, -9999) as pmin,
                                    coalesce(pmax, 9999) as pmax,
                                    flow
                                from
                                    {db_schema}.src_section
                                where
                                    version = {version}
                                    and date = '{tdate.isoformat()}'      
                            """,
            'supply': f"""
                        select
                            hour,
                            zone,
                            price,
                            volume
                        from
                            {db_schema}.src_supply
                        where
                            version = {version}
                            and date = '{tdate.isoformat()}'      
                    """,
            'demand': f"""
                            select
                                hour,
                                zone,
                                price,
                                volume
                            from
                                {db_schema}.src_demand
                            where
                                version = {version}
                                and date = '{tdate.isoformat()}'      
                        """
        },
        'topology_data': {
            'node': f"""
                        select
                            node,
                            type,
                            unom,
                            vzd,
                            gsh,
                            bsh,
                            pg,
                            pn,
                            qg,
                            qn,
                            qmin,
                            qmax,
                            umin,
                            umax
                        from
                            {db_schema}.dict_topo_node
                        where
                            version = {version}
                            and date = '{tdate.isoformat()}'
                    """,
            'vetv': f"""
                        select
                            node_from,
                            node_to,
                            pnum,
                            type,
                            r,
                            x,
                            g,
                            b,
                            b_from,
                            b_to,
                            ktr,
                            kti
                        from
                            {db_schema}.dict_topo_vetv
                        where
                            version = {version}
                            and date = '{tdate.isoformat()}'
                    """,
            'rge': f"""
                        select
                            rge,
                            type,
                            node,
                            fake_node
                        from
                            {db_schema}.dict_topo_rge
                        where
                            version = {version}
                            and date = '{tdate.isoformat()}'
                    """,
            'ges_opt': f"""
                            select
                                rge_group,
                                rge as rge_num,
                                type_opt
                            from
                                {db_schema}.dict_topo_opt_ges
                            where
                                version = {version}
                                and date = '{tdate.isoformat()}'
                        """,
            'section': f"""
                            select
                                sec_num,
                                node_from,
                                node_to,
                                dv,
                                forecast_type,
                                cz_id
                            from    
                                {db_schema}.dict_topo_section
                            where
                                version = {version}
                                and date = '{tdate.isoformat()}'
                        """,
            'node_geo': f"""
                                select
                                    node,
                                    cz_id,
                                    oes_id,
                                    zsp_id,
                                    sub_id
                                from
                                    {db_schema}.dict_node_geo
                                where
                                    version = {version}
                                    and date = '{tdate.isoformat()}'
                            """,
            'topo_gtp_con': f"""
                                select
                                    gtp_code,
                                    node,
                                    dist_coeff
                                from
                                    {db_schema}.dict_topo_gtp_con
                                where
                                    version = {version}
                                    and date = '{tdate.isoformat()}'
                            """
        }
    }

    return queries


def load_source_data(tdate: date, version: int, conn_string: str, db_schema: str):
    """
    Loads source data.
    :param tdate: target date
    :param version: target version
    :param conn_string: connection string to the database
    :param db_schema: database schema
    :return: loaded dataframes
    """

    engine = create_engine(conn_string)

    queries = generate_queries(tdate, version, db_schema)

    src_data = {}
    for k, v in queries['src_data'].items():
        src_data[k] = pd.read_sql(v, engine)

    topology_data = {}
    for k, v in queries['topology_data'].items():
        topology_data[k] = pd.read_sql(v, engine)

    return topology_data, src_data



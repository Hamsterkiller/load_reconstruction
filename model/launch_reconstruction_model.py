from datetime import date
from .data_loader import load_source_data
from .reconstruct_hour import reconstruct_hour


def launch_reconstruction_model(tdate: date, version: int, conn_string: str, db_schema: str):
    """
    Entry point to the model
    :param tdate: target date
    :param version: target version
    :param conn_string: connection string to the database
    :param db_schema: database schema
    :return: result datasets
    """

    # load source data
    topology_data, src_data = load_source_data(tdate, version, conn_string, db_schema)

    # fix rge-node conection for RGEs, which are associated t nodes with U = 110kV and below
    rge_node = topology_data['rge'].copy()
    rge_node['node'] = rge_node['fake_node'].combine_first(rge_node['node']).astype('int64')
    topology_data['rge'] = rge_node

    for t in range(0, 24):
        reconstruct_hour(t, topology_data, src_data)

    pass
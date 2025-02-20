import socket


def find_open_port(port: int) -> int:
    """Find a port not in use starting at given port"""
    # Taken from https://waylonwalker.com/python-find-available-port/, thanks Waylon!
    # https://github.com/WaylonWalker
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        if s.connect_ex(("localhost", port)) == 0:
            return find_open_port(port=port + 1)
        else:
            return port

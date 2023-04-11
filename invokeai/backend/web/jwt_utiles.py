import jwt
from typing import Any, Dict, List, Optional, Union

from pydantic import SecretStr


SecretType = Union[str, SecretStr]
JWT_ALGORITHM = "HS256"


def _get_secret_value(secret: SecretType) -> str:
    if isinstance(secret, SecretStr):
        return secret.get_secret_value()
    return secret


def decode_jwt(
    encoded_jwt: str,
    secret: SecretType,
    audience: List[str],
    algorithms: List[str] = [JWT_ALGORITHM],
) -> Dict[str, Any]:
    return jwt.decode(
        encoded_jwt,
        _get_secret_value(secret),
        audience=audience,
        algorithms=algorithms,
    )


def get_user_id_from_jwt(encoded_jwt: str) -> str:
    decoded_jwt = decode_jwt(
        encoded_jwt=encoded_jwt,
        secret="95bcb31e05fd4d4e949fa83f23246b4db1c0f1090725ad47d33e31ded2295494",
        audience=["fastapi-users:auth"],
        algorithms=["HS256"]
    )
    user_id = decoded_jwt.get("sub")
    return user_id
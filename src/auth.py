import secrets
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from src.config import settings

class AuthenticationError(HTTPException):
    def __init__(self, message: str = "Invalid or missing API key"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": message,
                    "type": "authentication_error",
                    "param": None,
                    "code": "invalid_api_key",
                }
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

def verify_api_key(request: Request) -> None:
    """
    Verify the Bearer token in the Authorization header.
    Clients must supply: 'Authorization: Bearer <API_KEY>'
    If API_KEY is configured on the server, all requests must present a valid match.
    """
    expected_key = settings.API_KEY.strip()
    
    # If an API_KEY is configured, strictly enforce authentication
    if expected_key:
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise AuthenticationError(
                "Missing Authorization header. Send 'Authorization: Bearer <API_KEY>'."
            )

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise AuthenticationError(
                "Invalid Authorization format. Expected 'Bearer <API_KEY>'."
            )

        token = parts[1]
        if not secrets.compare_digest(token, expected_key):
            raise AuthenticationError("Invalid API key provided.")

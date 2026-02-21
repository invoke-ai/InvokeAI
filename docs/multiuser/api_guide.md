# InvokeAI Multi-User API Guide

## Overview

This guide explains how to interact with InvokeAI's API in both single-user and multi-user modes. The API behavior depends on the `multiuser` configuration setting.

### Single-User vs Multi-User Mode

**Single-User Mode** (`multiuser: false` or option absent):
- No authentication required
- All API endpoints accessible without tokens
- Direct API access like previous InvokeAI versions
- All content visible in unified view

**Multi-User Mode** (`multiuser: true`):
- JWT token authentication required
- User-scoped access to resources
- Role-based authorization (admin vs regular user)
- Data isolation between users

## Authentication (Multi-User Mode Only)

### Authentication Flow

When multi-user mode is enabled, all API endpoints (except `/api/v1/auth/setup` and `/api/v1/auth/login`) require authentication using JWT (JSON Web Token) bearer tokens.

**Authentication Process:**

1. **Obtain Token**: POST credentials to `/api/v1/auth/login`
2. **Store Token**: Save the JWT token securely
3. **Use Token**: Include token in `Authorization` header for all requests
4. **Refresh**: Re-authenticate when token expires

!!! note "Single-User Mode"
    When running in single-user mode (`multiuser: false`), authentication endpoints are not available and authentication headers are not required.

### Login Endpoint

**Endpoint:** `POST /api/v1/auth/login`

**Request:**

```json
{
  "email": "user@example.com",
  "password": "SecurePassword123",
  "remember_me": false
}
```

**Response (Success):**

```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "user_id": "abc123",
    "email": "user@example.com",
    "display_name": "John Doe",
    "is_admin": false,
    "is_active": true,
    "created_at": "2024-01-15T10:00:00Z"
  },
  "expires_in": 86400
}
```

**Response (Error):**

```json
{
  "detail": "Incorrect email or password"
}
```

**Status Codes:**

- `200 OK` - Authentication successful
- `401 Unauthorized` - Invalid credentials
- `403 Forbidden` - Account disabled
- `422 Unprocessable Entity` - Invalid request format

### Using the Token

Include the JWT token in the `Authorization` header with the `Bearer` scheme:

**HTTP Header:**

```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Example HTTP Request:**

```http
GET /api/v1/boards HTTP/1.1
Host: localhost:9090
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json
```

### Token Expiration

Tokens have a limited lifetime:

- **Default**: 24 hours (86400 seconds)
- **Remember Me**: 7 days (604800 seconds)

**Handling Expiration:**

```python
import requests
import time

def api_request(url, token, max_retries=1):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 401:  # Token expired
        # Re-authenticate and retry
        new_token = login()
        headers = {"Authorization": f"Bearer {new_token}"}
        response = requests.get(url, headers=headers)
    
    return response
```

### Logout Endpoint

**Endpoint:** `POST /api/v1/auth/logout`

**Request:**

```http
POST /api/v1/auth/logout HTTP/1.1
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Response:**

```json
{
  "success": true
}
```

**Note:** With JWT tokens, logout is primarily client-side (delete token). Server-side session invalidation may be added in future releases.

## Code Examples

### Python

**Using `requests` library:**

```python
import requests
import json

class InvokeAIClient:
    def __init__(self, base_url="http://localhost:9090"):
        self.base_url = base_url
        self.token = None
        
    def login(self, email, password, remember_me=False):
        """Authenticate and store token."""
        url = f"{self.base_url}/api/v1/auth/login"
        payload = {
            "email": email,
            "password": password,
            "remember_me": remember_me
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        self.token = data["token"]
        return data["user"]
    
    def _get_headers(self):
        """Get headers with authentication token."""
        if not self.token:
            raise Exception("Not authenticated. Call login() first.")
        
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def get_boards(self):
        """Get user's boards."""
        url = f"{self.base_url}/api/v1/boards/"
        response = requests.get(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json()
    
    def create_board(self, board_name):
        """Create a new board."""
        url = f"{self.base_url}/api/v1/boards/"
        payload = {"board_name": board_name}
        
        response = requests.post(
            url, 
            json=payload,
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def logout(self):
        """Logout and clear token."""
        url = f"{self.base_url}/api/v1/auth/logout"
        response = requests.post(url, headers=self._get_headers())
        self.token = None
        return response.json()

# Usage
client = InvokeAIClient()
user = client.login("user@example.com", "SecurePassword123")
print(f"Logged in as: {user['display_name']}")

boards = client.get_boards()
print(f"User has {len(boards['items'])} boards")

new_board = client.create_board("My New Board")
print(f"Created board: {new_board['board_name']}")

client.logout()
```

**Error Handling:**

```python
import requests
from requests.exceptions import HTTPError

def safe_api_call(client, method, *args, **kwargs):
    """Make API call with error handling."""
    try:
        func = getattr(client, method)
        return func(*args, **kwargs)
    
    except HTTPError as e:
        if e.response.status_code == 401:
            print("Authentication failed or token expired")
            # Re-authenticate
            client.login(email, password)
            # Retry
            return func(*args, **kwargs)
        elif e.response.status_code == 403:
            print("Permission denied")
        elif e.response.status_code == 404:
            print("Resource not found")
        else:
            print(f"API error: {e.response.status_code}")
            print(e.response.text)
        
        raise

# Usage
try:
    boards = safe_api_call(client, "get_boards")
except Exception as e:
    print(f"Failed to get boards: {e}")
```

### JavaScript/TypeScript

**Using `fetch` API:**

```javascript
class InvokeAIClient {
  constructor(baseUrl = 'http://localhost:9090') {
    this.baseUrl = baseUrl;
    this.token = null;
  }

  async login(email, password, rememberMe = false) {
    const response = await fetch(`${this.baseUrl}/api/v1/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        email,
        password,
        remember_me: rememberMe,
      }),
    });

    if (!response.ok) {
      throw new Error(`Login failed: ${response.statusText}`);
    }

    const data = await response.json();
    this.token = data.token;
    
    // Store token in localStorage
    localStorage.setItem('invokeai_token', data.token);
    
    return data.user;
  }

  getHeaders() {
    if (!this.token) {
      throw new Error('Not authenticated. Call login() first.');
    }

    return {
      'Authorization': `Bearer ${this.token}`,
      'Content-Type': 'application/json',
    };
  }

  async getBoards() {
    const response = await fetch(`${this.baseUrl}/api/v1/boards/`, {
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to get boards: ${response.statusText}`);
    }

    return response.json();
  }

  async createBoard(boardName) {
    const response = await fetch(`${this.baseUrl}/api/v1/boards/`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ board_name: boardName }),
    });

    if (!response.ok) {
      throw new Error(`Failed to create board: ${response.statusText}`);
    }

    return response.json();
  }

  async logout() {
    const response = await fetch(`${this.baseUrl}/api/v1/auth/logout`, {
      method: 'POST',
      headers: this.getHeaders(),
    });

    this.token = null;
    localStorage.removeItem('invokeai_token');

    return response.json();
  }
}

// Usage
(async () => {
  const client = new InvokeAIClient();
  
  try {
    const user = await client.login('user@example.com', 'SecurePassword123');
    console.log(`Logged in as: ${user.display_name}`);
    
    const boards = await client.getBoards();
    console.log(`User has ${boards.items.length} boards`);
    
    const newBoard = await client.createBoard('My New Board');
    console.log(`Created board: ${newBoard.board_name}`);
    
    await client.logout();
  } catch (error) {
    console.error('Error:', error.message);
  }
})();
```

**TypeScript with Types:**

```typescript
interface LoginRequest {
  email: string;
  password: string;
  remember_me?: boolean;
}

interface User {
  user_id: string;
  email: string;
  display_name: string;
  is_admin: boolean;
  is_active: boolean;
  created_at: string;
}

interface LoginResponse {
  token: string;
  user: User;
  expires_in: number;
}

interface Board {
  board_id: string;
  board_name: string;
  created_at: string;
  updated_at: string;
  deleted_at?: string;
  cover_image_name?: string;
}

class InvokeAIClient {
  private baseUrl: string;
  private token: string | null = null;

  constructor(baseUrl: string = 'http://localhost:9090') {
    this.baseUrl = baseUrl;
  }

  async login(
    email: string, 
    password: string, 
    rememberMe: boolean = false
  ): Promise<User> {
    const response = await fetch(`${this.baseUrl}/api/v1/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, remember_me: rememberMe }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Login failed');
    }

    const data: LoginResponse = await response.json();
    this.token = data.token;
    return data.user;
  }

  private getHeaders(): HeadersInit {
    if (!this.token) {
      throw new Error('Not authenticated');
    }
    return {
      'Authorization': `Bearer ${this.token}`,
      'Content-Type': 'application/json',
    };
  }

  async getBoards(): Promise<{ items: Board[] }> {
    const response = await fetch(`${this.baseUrl}/api/v1/boards/`, {
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error('Failed to get boards');
    }

    return response.json();
  }
}
```

### cURL

**Login:**

```bash
# Login and extract token
TOKEN=$(curl -X POST http://localhost:9090/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePassword123",
    "remember_me": false
  }' | jq -r '.token')

echo "Token: $TOKEN"
```

**Get Boards:**

```bash
curl -X GET http://localhost:9090/api/v1/boards/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json"
```

**Create Board:**

```bash
curl -X POST http://localhost:9090/api/v1/boards/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "board_name": "My API Board"
  }'
```

**Generate Image:**

```bash
curl -X POST http://localhost:9090/api/v1/sessions/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape",
    "width": 512,
    "height": 512,
    "steps": 30
  }'
```

## API Endpoint Changes

### Authentication Required

All endpoints now require authentication except:

- `POST /api/v1/auth/setup` - Initial admin setup
- `POST /api/v1/auth/login` - User login

### User-Scoped Resources

Resources are now filtered by the authenticated user:

**Boards:**

```python
# Before (single-user)
GET /api/v1/boards/  # Returns all boards

# After (multi-user)
GET /api/v1/boards/  # Returns only current user's boards
```

**Images:**

```python
# Images are filtered by board ownership
GET /api/v1/images/  # Only shows images on user's boards
```

**Workflows:**

```python
# Returns user's workflows + public workflows
GET /api/v1/workflows/
```

**Queue:**

```python
# Regular users see only their queue items
GET /api/v1/queue/  # User's queue items

# Administrators see all queue items
GET /api/v1/queue/  # All users' queue items
```

### Administrator Endpoints

Some endpoints require administrator privileges:

**User Management:**

```python
GET    /api/v1/users           # List users (admin only)
POST   /api/v1/users           # Create user (admin only)
GET    /api/v1/users/{id}      # Get user (admin only)
PATCH  /api/v1/users/{id}      # Update user (admin only)
DELETE /api/v1/users/{id}      # Delete user (admin only)
```

**Model Management (Write Operations):**

```python
POST   /api/v1/models/install          # Install model (admin only)
DELETE /api/v1/models/i/{key}          # Delete model (admin only)
PATCH  /api/v1/models/i/{key}          # Update model (admin only)
PUT    /api/v1/models/convert/{key}    # Convert model (admin only)
```

**Model Management (Read Operations):**

```python
GET /api/v1/models/                    # List models (all users)
GET /api/v1/models/i/{key}             # Get model details (all users)
```

### Error Responses

**401 Unauthorized:**

```json
{
  "detail": "Invalid authentication credentials"
}
```

Occurs when:

- Token is missing
- Token is invalid
- Token is expired
- Token signature is invalid

**403 Forbidden:**

```json
{
  "detail": "Admin privileges required"
}
```

Occurs when:

- User attempts admin-only operation
- Account is disabled
- Insufficient permissions

**404 Not Found:**

```json
{
  "detail": "Resource not found"
}
```

Occurs when:

- Resource doesn't exist
- User doesn't have access to resource

## New API Endpoints

### Authentication Endpoints

#### Setup Administrator

**Endpoint:** `POST /api/v1/auth/setup`

**Description:** Create initial administrator account (only works if no admin exists)

**Request:**

```json
{
  "email": "admin@example.com",
  "display_name": "Administrator",
  "password": "SecureAdminPass123"
}
```

**Response:**

```json
{
  "success": true,
  "user": {
    "user_id": "abc123",
    "email": "admin@example.com",
    "display_name": "Administrator",
    "is_admin": true,
    "is_active": true
  }
}
```

#### Get Current User

**Endpoint:** `GET /api/v1/auth/me`

**Description:** Get currently authenticated user's information

**Request:**

```http
GET /api/v1/auth/me
Authorization: Bearer <token>
```

**Response:**

```json
{
  "user_id": "abc123",
  "email": "user@example.com",
  "display_name": "John Doe",
  "is_admin": false,
  "is_active": true,
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:00:00Z",
  "last_login_at": "2024-01-15T15:30:00Z"
}
```

#### Change Password

**Endpoint:** `POST /api/v1/auth/change-password`

**Description:** Change current user's password

**Request:**

```json
{
  "current_password": "OldPassword123",
  "new_password": "NewPassword456"
}
```

**Response:**

```json
{
  "success": true
}
```

### User Management Endpoints (Admin Only)

#### List Users

**Endpoint:** `GET /api/v1/users`

**Request:**

```http
GET /api/v1/users?page=1&per_page=20
Authorization: Bearer <admin_token>
```

**Response:**

```json
{
  "items": [
    {
      "user_id": "abc123",
      "email": "user@example.com",
      "display_name": "John Doe",
      "is_admin": false,
      "is_active": true,
      "created_at": "2024-01-15T10:00:00Z",
      "last_login_at": "2024-01-15T15:30:00Z"
    }
  ],
  "page": 1,
  "pages": 1,
  "per_page": 20,
  "total": 5
}
```

#### Create User

**Endpoint:** `POST /api/v1/users`

**Request:**

```json
{
  "email": "newuser@example.com",
  "display_name": "New User",
  "password": "TempPassword123",
  "is_admin": false
}
```

**Response:**

```json
{
  "user_id": "xyz789",
  "email": "newuser@example.com",
  "display_name": "New User",
  "is_admin": false,
  "is_active": true,
  "created_at": "2024-01-15T16:00:00Z"
}
```

#### Update User

**Endpoint:** `PATCH /api/v1/users/{user_id}`

**Request:**

```json
{
  "display_name": "Updated Name",
  "is_active": true,
  "is_admin": false
}
```

**Response:**

```json
{
  "user_id": "xyz789",
  "email": "newuser@example.com",
  "display_name": "Updated Name",
  "is_admin": false,
  "is_active": true
}
```

#### Delete User

**Endpoint:** `DELETE /api/v1/users/{user_id}`

**Response:**

```json
{
  "success": true
}
```

#### Reset User Password

**Endpoint:** `POST /api/v1/users/{user_id}/reset-password`

**Request:**

```json
{
  "new_password": "NewTempPass123"
}
```

**Response:**

```json
{
  "success": true
}
```

### Board Sharing Endpoints

#### Share Board

**Endpoint:** `POST /api/v1/boards/{board_id}/share`

**Request:**

```json
{
  "user_id": "user123",
  "permission": "write"
}
```

**Response:**

```json
{
  "success": true,
  "share": {
    "board_id": "board456",
    "user_id": "user123",
    "permission": "write",
    "shared_at": "2024-01-15T16:00:00Z"
  }
}
```

#### List Board Shares

**Endpoint:** `GET /api/v1/boards/{board_id}/shares`

**Response:**

```json
{
  "items": [
    {
      "user_id": "user123",
      "display_name": "John Doe",
      "permission": "write",
      "shared_at": "2024-01-15T16:00:00Z"
    }
  ]
}
```

#### Remove Board Share

**Endpoint:** `DELETE /api/v1/boards/{board_id}/share/{user_id}`

**Response:**

```json
{
  "success": true
}
```

## Best Practices

### Token Storage

**Do:**

- Store tokens securely (keychain, secure storage)
- Use HTTPS to transmit tokens
- Clear tokens on logout
- Handle token expiration gracefully

**Don't:**

- Store tokens in URL parameters
- Log tokens in plain text
- Share tokens between users
- Store tokens in version control

### Error Handling

Always handle authentication errors:

```python
def make_request(client, func, *args, **kwargs):
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            return func(*args, **kwargs)
        except AuthenticationError:
            if retry_count >= max_retries - 1:
                raise
            # Re-authenticate
            client.login(email, password)
            retry_count += 1
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
```

### Rate Limiting

Be mindful of API rate limits:

- Implement exponential backoff for retries
- Cache frequently accessed data
- Batch requests when possible
- Don't hammer the login endpoint

### Connection Management

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_session():
    """Create session with retry logic."""
    session = requests.Session()
    
    retry = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    return session
```

## Migration Guide

### Updating Existing Code

**Before (single-user mode):**

```python
import requests

def get_boards():
    response = requests.get("http://localhost:9090/api/v1/boards/")
    return response.json()
```

**After (multi-user mode):**

```python
import requests

class APIClient:
    def __init__(self):
        self.token = None
    
    def login(self, email, password):
        response = requests.post(
            "http://localhost:9090/api/v1/auth/login",
            json={"email": email, "password": password}
        )
        self.token = response.json()["token"]
    
    def get_boards(self):
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(
            "http://localhost:9090/api/v1/boards/",
            headers=headers
        )
        return response.json()

# Usage
client = APIClient()
client.login("user@example.com", "password")
boards = client.get_boards()
```

### Backward Compatibility

InvokeAI supports both single-user and multi-user modes via the `multiuser` configuration option.

**Configuration:**

```yaml
# invokeai.yaml

# Single-user mode (no authentication)
multiuser: false  # or omit the option entirely

# Multi-user mode (authentication required)
multiuser: true
```

**Checking Mode Programmatically:**

```python
def is_multiuser_enabled(base_url):
    """Check if multi-user mode is enabled (authentication required)."""
    response = requests.get(f"{base_url}/api/v1/boards/")
    return response.status_code == 401  # 401 = auth required

# Example usage
base_url = "http://localhost:9090"
if is_multiuser_enabled(base_url):
    print("Multi-user mode: authentication required")
    # Use authenticated API calls
else:
    print("Single-user mode: no authentication needed")
    # Use direct API calls
```

**Adaptive Client:**

```python
class AdaptiveInvokeAIClient:
    def __init__(self, base_url="http://localhost:9090"):
        self.base_url = base_url
        self.token = None
        self.multiuser_mode = self._check_multiuser_mode()
    
    def _check_multiuser_mode(self):
        """Detect if multi-user mode is enabled."""
        try:
            response = requests.get(f"{self.base_url}/api/v1/boards/")
            return response.status_code == 401
        except:
            return False
    
    def login(self, email, password):
        """Login (only needed in multi-user mode)."""
        if not self.multiuser_mode:
            print("Single-user mode: login not required")
            return
        
        response = requests.post(
            f"{self.base_url}/api/v1/auth/login",
            json={"email": email, "password": password}
        )
        self.token = response.json()["token"]
    
    def _get_headers(self):
        """Get headers (with auth token if in multi-user mode)."""
        if self.multiuser_mode and self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}
    
    def get_boards(self):
        """Get boards (works in both modes)."""
        response = requests.get(
            f"{self.base_url}/api/v1/boards/",
            headers=self._get_headers()
        )
        return response.json()
```

## OpenAPI/Swagger Documentation

InvokeAI provides OpenAPI documentation for all endpoints.

**Access Swagger UI:**

```
http://localhost:9090/docs
```

**Download OpenAPI Schema:**

```bash
curl http://localhost:9090/openapi.json > invokeai_openapi.json
```

**Generate Client Code:**

Use tools like `openapi-generator` to generate client libraries:

```bash
# Generate Python client
openapi-generator generate \
  -i http://localhost:9090/openapi.json \
  -g python \
  -o ./invokeai-client

# Generate TypeScript client
openapi-generator generate \
  -i http://localhost:9090/openapi.json \
  -g typescript-fetch \
  -o ./invokeai-client-ts
```

## Security Considerations

### HTTPS

Always use HTTPS in production:

```python
# Development
client = InvokeAIClient("http://localhost:9090")

# Production
client = InvokeAIClient("https://invoke.example.com")
```

### Token Security

Protect JWT tokens:

```python
# Never log tokens
logger.info(f"User logged in")  # Good
logger.info(f"Token: {token}")  # Bad!

# Use environment variables for credentials
import os
email = os.environ.get("INVOKEAI_EMAIL")
password = os.environ.get("INVOKEAI_PASSWORD")
```

### Input Validation

Always validate user input:

```python
import re

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Check password meets requirements."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not any(c.isupper() for c in password):
        return False, "Password must contain uppercase letters"
    if not any(c.islower() for c in password):
        return False, "Password must contain lowercase letters"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain numbers"
    return True, ""
```

## Troubleshooting

### Common Issues

**Issue: "Invalid authentication credentials"**

- Token expired - re-authenticate
- Token malformed - check token string
- Token signature invalid - check secret key hasn't changed

**Issue: "Admin privileges required"**

- User is not an administrator
- Use admin account for this operation

**Issue: Token not being sent**

- Check `Authorization` header is present
- Verify `Bearer` prefix is included
- Check token isn't truncated

**Issue: CORS errors**

Configure CORS in InvokeAI:

```yaml
# invokeai.yaml
cors_origins:
  - "http://localhost:3000"
  - "https://myapp.example.com"
```

## Additional Resources

- [User Guide](user_guide.md) - For end users
- [Administrator Guide](admin_guide.md) - For administrators
- [Multiuser Specification](specification.md) - Technical details
- [OpenAPI Documentation](http://localhost:9090/docs) - Interactive API docs
- [GitHub Repository](https://github.com/invoke-ai/InvokeAI) - Source code

---

**Questions?** Visit the [InvokeAI Discord](https://discord.gg/ZmtBAhwWhy) or check the [FAQ](../faq.md).

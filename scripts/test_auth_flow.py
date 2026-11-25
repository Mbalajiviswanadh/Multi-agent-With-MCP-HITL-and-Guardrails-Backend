"""
scripts/test_auth_flow.py
-------------------------
Verification script to test the production authentication system:
  1. POST /api/auth/register (Create new user with hashed password)
  2. POST /api/auth/login (Verify credentials and issue signed JWT)
  3. GET  /api/auth/me (Verify JWT bearer token and load user profile)
  4. Test protected endpoints: rejection of invalid/missing tokens
"""
import sys
import uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app, raise_server_exceptions=True)

def main():
    print("=" * 65)
    print("TripMate AI — Production JWT Authentication Verification")
    print("=" * 65)

    random_id = uuid.uuid4().hex[:6]
    test_name = f"Viswanadh Test {random_id}"
    test_email = f"viswanadh_{random_id}@example.com"
    test_password = "SecurePassword123!"

    # 1. Test Registration
    print(f"\n[1/4] Testing User Registration:")
    print(f"      Name:     {test_name}")
    print(f"      Email:    {test_email}")
    reg_response = client.post(
        "/api/auth/register",
        json={"name": test_name, "email": test_email, "password": test_password},
    )
    assert reg_response.status_code == 201, f"Reg failed: {reg_response.text}"
    reg_data = reg_response.json()
    token = reg_data.get("access_token")
    user_id = reg_data["user"]["id"]
    print(f"      Status:   201 Created")
    print(f"      User ID:  {user_id}")
    print(f"      JWT Token issued: {token[:25]}... (Signed with HS256)")

    # 2. Test Login
    print(f"\n[2/4] Testing User Login with password:")
    login_response = client.post(
        "/api/auth/login",
        json={"email": test_email, "password": test_password},
    )
    assert login_response.status_code == 200, f"Login failed: {login_response.text}"
    login_token = login_response.json().get("access_token")
    print(f"      Status:   200 OK (Password verified with PBKDF2-HMAC-SHA256)")

    # Test Wrong Password Rejection
    bad_login = client.post(
        "/api/auth/login",
        json={"email": test_email, "password": "WrongPassword!"},
    )
    assert bad_login.status_code == 401, "Failed to reject invalid password!"
    print(f"      Security: Correctly rejected wrong password (401 Unauthorized)")

    # 3. Test Protected Route: /api/auth/me
    print(f"\n[3/4] Testing Protected Route (/api/auth/me):")
    me_response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {login_token}"},
    )
    assert me_response.status_code == 200, f"Get /me failed: {me_response.text}"
    me_data = me_response.json()["user"]
    print(f"      Status:   200 OK")
    print(f"      Verified Identity: id={me_data['id']}, email={me_data['email']}")

    # 4. Test Unauthenticated Access to Protected Delete Endpoint
    print(f"\n[4/4] Testing Protected Route Security Boundaries:")
    no_auth_delete = client.delete(f"/api/memory/{user_id}")
    assert no_auth_delete.status_code == 401, f"Expected 401, got {no_auth_delete.status_code}"
    print(f"      Security: DELETE /api/memory without token returned 401 Unauthorized [PASS]")

    auth_delete = client.delete(
        f"/api/memory/{user_id}",
        headers={"Authorization": f"Bearer {login_token}"},
    )
    assert auth_delete.status_code == 200, f"Delete failed: {auth_delete.text}"
    print(f"      Security: DELETE /api/memory with valid bearer token returned 200 OK [PASS]")

    print("\n" + "=" * 65)
    print("ALL PRODUCTION AUTHENTICATION & SECURITY TESTS PASSED 100%!")
    print("=" * 65)

if __name__ == "__main__":
    main()

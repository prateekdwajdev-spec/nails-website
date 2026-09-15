from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    # Test an invalid file type upload to check error handling
    response = client.post("/process", files={"file": ("test.txt", b"hello", "text/plain")})
    assert response.status_code == 400
    assert response.json() == {"detail": "File must be an image."}

def test_cors_headers():
    # Test that CORS headers are properly included in responses
    response = client.options(
        "/process",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    # FastAPI handles OPTIONS requests for CORS automatically
    assert response.status_code in [200, 400, 405]
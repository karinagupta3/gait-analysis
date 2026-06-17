"""Tests for the app-level shared-password authentication helpers.

These exercise the signed-cookie session logic directly (no FastAPI needed), so
they run in the local suite even though test_web.py is skipped without fastapi.
"""

import hashlib
import hmac
import importlib
import time


def _reload_with_pw(pw, key="unit-test-key"):
    import os
    os.environ["GAIT_AUTH_PASSWORD"] = pw
    os.environ["GAIT_SECRET_KEY"] = key
    from gait_analysis.web import app as A
    return importlib.reload(A)


def test_auth_disabled_without_password(monkeypatch):
    monkeypatch.delenv("GAIT_AUTH_PASSWORD", raising=False)
    from gait_analysis.web import app as A
    importlib.reload(A)
    assert A._auth_enabled() is False


def test_valid_token_accepted_tampered_rejected():
    A = _reload_with_pw("pw")
    assert A._auth_enabled() is True
    tok = A._make_session_token()
    assert A._valid_session(tok)
    flipped = tok[:-1] + ("0" if tok[-1] != "0" else "1")
    assert not A._valid_session(flipped)
    assert not A._valid_session("garbage")
    assert not A._valid_session("")


def test_expired_token_rejected():
    A = _reload_with_pw("pw")
    exp = str(int(time.time()) - 5)
    sig = hmac.new(A.AUTH_SECRET, exp.encode(), hashlib.sha256).hexdigest()
    assert not A._valid_session(f"{exp}.{sig}")


def test_token_signed_with_different_key_rejected():
    A = _reload_with_pw("pw", key="key-one")
    tok = A._make_session_token()
    A2 = _reload_with_pw("pw", key="key-two")
    assert not A2._valid_session(tok)  # signature won't verify under the new key

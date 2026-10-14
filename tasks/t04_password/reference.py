def validate_password(pw):
    return (
        len(pw) >= 8
        and any(c.isupper() for c in pw)
        and any(c.islower() for c in pw)
        and any(c.isdigit() for c in pw)
        and any(c in "!@#$%" for c in pw)
        and " " not in pw
    )

from slr.slr_orbit_validation import SLROrbitValidation


if __name__ == '__main__':
    orbit_validation = SLROrbitValidation()
    # Residual (difference between the measured range and the real distance) vs. transmission time
    orbit_validation.first_orbit_validation()

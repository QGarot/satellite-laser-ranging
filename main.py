from slr.slr_orbit_validation import SLROrbitValidation

if __name__ == '__main__':
    orbit_validation = SLROrbitValidation()
    # Residuals histogram
    orbit_validation.second_orbit_validation(debug=True)

# Changelog

## [1.02] – 2025-MM-DD  

### Added
- `start_MiPAS.bat` for easier application launch on Windows.
- `INSTALLATION_GUIDE_MiPAS.md`

### Moved
- Moved `CHANGELOG.md` and `license.txt` into the `docs/` directory for better project organization.

### Fixed
- Typo in `linear_entropy_analysis.py` – corrected label capitalization in plot:

  ```
  plt.plot(x_positions, sliding_window_entropy, label="sliding Window Entropy")
  → plt.plot(x_positions, sliding_window_entropy, label="Sliding Window Entropy")
  ```

- Incorrect import and base class in `analysis_template.py`:

  ```
  from mipas.config.base_configuration_manager import BaseConfigurationManager
  → from mipas.config.configuration_manager import ConfigurationManager

  class CustomAnalysisConfigurationManager(BaseConfigurationManager):
  → class CustomAnalysisConfigurationManager(ConfigurationManager):
  ```

### Cleaned
- Removed or clarified outdated comments across several modules.

---

## [1.01] – 2025-05-06  
*Initial public release.*

### Added
- Initial version of MiPAS released publicly.

# Hill of Towie SCADA Data Audit

The audit covers all 124 months from January 2016 through April 2026 and all 21 turbines in Zenodo release 2.1.0. It uses the ten-minute turbine and grid tables for active power, active wind speed, generator RPM, and the primary, corrected secondary, and raw secondary wind-speed channels. [Source release](https://zenodo.org/records/22662930) · [SCADA data dictionary](https://zenodo.org/records/22662930/preview/Hill_of_Towie_data_dictionary.md?include_deleted=0)

## Missingness

At least one forecast input was unavailable in **166,440 of 11,409,552 expected turbine intervals (1.46%)**. Both source tables lacked the turbine and timestamp in 165,328 of those intervals, or 99.33% of incomplete intervals. Per channel, 166,433 active-power, 166,431 active-wind-speed, and 166,430 RPM measurements were unavailable. The respective counts of blank values on present rows were 1,101, 1,102, and 1,101.

The window audit uses the connector's hourly cutoffs within each UTC source month. Each usable window requires 36 consecutive historical readings for all three inputs and six future power readings. The audit recovered **1,842,612 of 1,885,968 candidate windows (97.70%)**, including exactly the 59,471 windows in the existing 2026-only connector.

| Year | Expected turbine intervals | Incomplete input intervals | Incomplete | Usable windows | Usable |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2016 | 1,106,784 | 9,039 | 0.82% | 179,462 | 98.09% |
| 2017 | 1,103,760 | 15,916 | 1.44% | 177,546 | 97.31% |
| 2018 | 1,103,760 | 9,029 | 0.82% | 178,586 | 97.88% |
| 2019 | 1,103,760 | 3,127 | 0.28% | 180,977 | 99.19% |
| 2020 | 1,106,784 | 5,511 | 0.50% | 181,338 | 99.12% |
| 2021 | 1,103,760 | 12,978 | 1.18% | 179,335 | 98.29% |
| 2022 | 1,103,760 | 55,791 | 5.05% | 170,948 | 93.70% |
| 2023 | 1,103,760 | 5,355 | 0.49% | 180,554 | 98.96% |
| 2024 | 1,106,784 | 3,919 | 0.35% | 180,967 | 98.92% |
| 2025 | 1,103,760 | 44,368 | 4.02% | 173,428 | 95.06% |
| 2026 through April | 362,880 | 1,407 | 0.39% | 59,471 | 99.16% |

- September 2025 was the worst month for fleet coverage. **44.34%** of expected turbine intervals lacked an input. Both tables had zero turbine rows for 1,833 consecutive ten-minute intervals, with end timestamps from September 10 at 18:20 to September 23 at 11:40 UTC. That is **12.73 days** without SCADA rows for any turbine.
- August and September 2022 were missing **22.00%** and **13.76%**, respectively. Several turbines had roughly 20 consecutive days of missing input in August.
- T20 had the longest individual gap, **6,969 intervals (48.40 days)** from March 24 to May 12, 2021. T06 was missing **6,949 intervals (48.26 days)** from January 21 to March 10, 2022.
- The tables contained 5,184 rows outside their owning month, which the audit excluded using the publisher's interval-start convention. Within the owning months there were zero duplicate turbine/timestamp keys, invalid or off-grid timestamps, unknown station IDs, or infinite values in the audited fields. [RES SCADA loader](https://github.com/resgroup/hill-of-towie-open-source-analysis/blob/main/src/hot_open/scada_helpers.py)

## Wind sensor changes

For the documented 2017–2018 correction, the audit compared the 30 days on either side of each turbine's correction date and excluded the change date. The comparison used simultaneous readings in four 2 m/s primary-wind bands from 4 to 12 m/s. It compared medians within each turbine and band, averaged matched bands within each turbine, and took the median change across turbines. All 21 turbines met the minimum of 50 readings per side and band for at least one match. [Correction dates and channel definitions](https://zenodo.org/records/22662930/preview/Hill_of_Towie_data_dictionary.md?include_deleted=0)

| Paired comparison | Median change across turbines |
| --- | ---: |
| Raw secondary / primary around the correction | +11.91 percentage points |
| Corrected secondary / primary around the correction | −0.97 points |
| Active / primary around the correction | 0.00 points |
| Corrected secondary / primary, Jan–Sep 2019–2020 to Jan–Sep 2025 | +1.49 points |
| Corrected secondary / primary, Jan–Apr 2025 to Jan–Apr 2026 | −0.20 points |

The longer comparisons match calendar month as well as turbine and wind-speed band. Their fleet medians hide individual shifts. T05 moved **+5.44 points** in the 2019–2020 to 2025 comparison. From January–April 2025 to the same months in 2026, T02 and T03 shifted **−9.28** and **−7.27 points**, respectively, in corrected secondary relative to primary. Across all wind speeds, **99.75%** of readings uniquely matching one of the two onboard channels within 0.01 m/s matched the primary. That comparison used 11,002,700 unique matches.

For an external check, I matched 1,052,504 complete hourly turbine records to the hourly 100 m ERA5 grid point at 57.50° N, 3.25° W. I compared active-wind-to-ERA5 ratios within each turbine, calendar month, wind-direction quadrant, and 2 m/s ERA5 speed band from 6 to 14 m/s. Against the matched 2019–2020 baseline, the median annual fleet shifts ranged from **−2.32 points in 2016 to +2.17 points in 2023**. The 2025 shift was **+1.98 points** and January–April 2026 was **−0.45 points** against the corresponding baseline months. ERA5 estimates grid-scale wind above hub height. These ratios capture changes in the local-to-grid relationship, including possible sensor drift, changes in wakes, and other operating effects. An absolute calibration estimate would require a trusted local reference. [ERA5 documentation](https://confluence.ecmwf.int/pages/viewpage.action?pageId=179741903) · [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)

## Effect on the forecasting split

The proposed October–December 2024 validation period keeps **45,511 of 45,990** possible windows (98.96%). The main 2025 test through October 28 at 13:00 UTC keeps **141,619 of 150,213** (94.28%). September accounts for **80.6%** of the main test's lost windows. Across the other main-test months, **98.77%** of windows are usable. Keep September's available windows in the primary score and report the coverage and a sensitivity score with September omitted.

Keep the historical active-wind channel for the full-history training run. The raw-secondary correction has little bearing on that input, and the measured gaps are concentrated in identifiable periods. Compare full-history training with the post-AeroUp history on the same validation set to see whether the additional years improve forecast accuracy.

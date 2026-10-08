# Quotes from the shipped pricer

Produced by `scripts/run_all.py` through the same `quote()` function the app calls.

## new driver, city, powerful car

Driver 19, bonus-malus 100, vehicle age 2, power band 9, brand B12, regular fuel, 9,000 people per km2, Ile-de-France.

- shipped model: office premium EUR 679 (technical EUR 475, 0.1907 claims a year)
- best GLM (with interactions): office premium EUR 963 (technical EUR 674, 0.2709 claims a year)

Biggest factors in the shipped price:

- Bonus-malus level 100: 198% dearer than an average policy
- Vehicle brand B12: 23% cheaper than an average policy
- Vehicle power band 9: 15% dearer than an average policy
- Driver age 19: 13% dearer than an average policy

## experienced driver, small town

Driver 47, bonus-malus 50, vehicle age 6, power band 6, brand B1, diesel fuel, 300 people per km2, Bretagne.

- shipped model: office premium EUR 190 (technical EUR 133, 0.0594 claims a year)
- best GLM (with interactions): office premium EUR 214 (technical EUR 150, 0.0638 claims a year)

Biggest factors in the shipped price:

- Bonus-malus level 50: 34% cheaper than an average policy
- Driver age 47: 19% dearer than an average policy
- Region Bretagne: 8% cheaper than an average policy
- Fuel Diesel: 9% dearer than an average policy

## driver with a claims record

Driver 35, bonus-malus 95, vehicle age 10, power band 7, brand B2, diesel fuel, 2,500 people per km2, Provence-Alpes-Cotes-D'Azur.

- shipped model: office premium EUR 567 (technical EUR 397, 0.1540 claims a year)
- best GLM (with interactions): office premium EUR 547 (technical EUR 383, 0.1577 claims a year)

Biggest factors in the shipped price:

- Bonus-malus level 95: 112% dearer than an average policy
- Population density 2,500 people per km2: 13% dearer than an average policy
- Driver age 35: 10% cheaper than an average policy
- Fuel Diesel: 6% dearer than an average policy

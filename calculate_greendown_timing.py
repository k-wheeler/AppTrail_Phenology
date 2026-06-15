import ee


def compute_greendown_date(hls_evi, route_buffer, year):
    hls_evi = hls_evi.select('EVI')

    # Per-pixel seasonal means
    july_mean = hls_evi.filterDate(f'{year}-07-01', f'{year}-07-31').mean()
    dec_mean  = hls_evi.filterDate(f'{year}-12-01', f'{year}-12-31').mean()

    # Per-pixel threshold: July mean minus 75% of seasonal decline
    threshold = july_mean.subtract(july_mean.subtract(dec_mean).multiply(0.75))

    # Late-season collection with DOY band
    late_season = (
        hls_evi
        .filterDate(f'{year}-07-01', f'{year}-12-31')
        .sort('system:time_start')
    )

    def add_doy(img):
        doy = ee.Date(img.get('system:time_start')).getRelative('day', 'year').add(1)
        return img.addBands(ee.Image.constant(doy).toFloat().rename('DOY'))

    def apply_mask(img):
        below_threshold = img.select('EVI').lte(threshold)
        return img.select('DOY').updateMask(below_threshold)

    masked = late_season.map(add_doy).map(apply_mask)

    return masked.min().rename('Greendown_DOY').clip(route_buffer)

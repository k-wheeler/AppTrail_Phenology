import ee


def compute_hls_evi(route_buffer, ma_forest, year):
    def add_evi(img):
        evi = img.expression(
            '2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))',
            {
                'nir': img.select('B5'),
                'red': img.select('B4'),
                'blue': img.select('B2'),
            }
        ).rename('EVI')
        return evi.copyProperties(img, ['system:time_start'])

    return (
        ee.ImageCollection("NASA/HLS/HLSL30/v002")
        .filterBounds(route_buffer)
        .filterDate(f'{year}-07-01', f'{year}-12-31')
        .select(['B5', 'B4', 'B2'])
        .map(add_evi)
    )

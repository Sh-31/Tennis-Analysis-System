def convert_pixel_distance_to_meter(pixel_distance , ref_heigth_meters, ref_heigth_pixel):
    # 20 --> 10.97
    # pixel_dist --> x
    return ((pixel_distance * ref_heigth_meters) / ref_heigth_pixel)

def covert_meters_to_pixal_distance(meters, ref_heigth_meters , ref_heigth_pixel):
    return ((meters * ref_heigth_pixel) / ref_heigth_meters)




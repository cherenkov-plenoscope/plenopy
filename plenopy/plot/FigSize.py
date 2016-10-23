class FigSize(object):
    def __init__(
        self, 
        relative_width=16, 
        relative_hight=9, 
        pixel_rows=1080, 
        dpi=200):

        self.relative_width = relative_width
        self.relative_hight = relative_hight

        self.dpi = dpi

        self.pixel_rows = int(pixel_rows)
        self.pixel_cols = int(pixel_rows*(relative_width/relative_hight))

        self.hight = pixel_rows/dpi
        self.width = self.hight*(relative_width/relative_hight)
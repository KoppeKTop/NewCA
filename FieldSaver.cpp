// Field saver utilites
#include "FieldSaver.h"
#include "genca.h"
#include "EasyBMP.h"

RGBApixel* get_colours_table()
{
	RGBApixel * res = new RGBApixel[LABEL_LAST];

	res[LABEL_EMPTY].Red = 255;
	res[LABEL_EMPTY].Green = 255;
	res[LABEL_EMPTY].Blue = 255;
	res[LABEL_EMPTY].Alpha = 255;
	
	res[LABEL_AG].Red = 0;
	res[LABEL_AG].Green = 0;
	res[LABEL_AG].Blue = 0;
	res[LABEL_AG].Alpha = 0;
	
	res[LABEL_DRUG].Red = 255;
	res[LABEL_DRUG].Green = 0;
	res[LABEL_DRUG].Blue = 0;
	res[LABEL_DRUG].Alpha = 0;
	
	return res;
}

extern "C" int save_bmp(const ElementType * fld, const Coord sz, const char * filename)
{
//	Coord sz = fld->GetSize();
	BMP * pic = new BMP();
	pic->SetSize(sz.GetCoord(0), sz.GetCoord(1));
	RGBApixel * tbl = get_colours_table();
	pic->SetBitDepth(4);
	for (int i = 0; i < sz.GetCoord(0); ++i)
	{
		for (int j = 0; j < sz.GetCoord(1); ++j)
		{
			FieldElement lbl = fld[i + sz.GetCoord(0) * j];
			if (0 <= lbl && lbl < LABEL_LAST)
			{
				RGBApixel curr_colour = tbl[lbl];
				pic->SetPixel(i, j, curr_colour);
			}
			else
			{
				fprintf(stderr, "Wrong label to draw: %u!\n", lbl);
			}
		}
	}
	pic->WriteToFile(filename);
	delete pic;
	delete [] tbl;
	return 0;
}


#include "newca.h"
#include <stdio.h>
#include "genca.h"
#include <fstream>
#include "Coord.h"
#include "CellsField.h"
#include <algorithm>
#include "load_params.h"

#include <stdlib.h>
#include <string.h>

#ifndef __cplusplus
#define __cplusplus
#endif
#include "libini.h"
using namespace std;
//
//extern "C" char * ini_readString(dictionary * d, const char * key, char * def);
//extern "C" int ini_readInt(dictionary * d, const char * key, int notfound);
//extern "C" double ini_readDouble(dictionary * d, char * key, double notfound);
//extern "C" dictionary * iniparser_load(const char * ininame);
//extern "C" void iniparser_freedict(dictionary * d);

const int NOT_FOUND_VAL = 9000;

#define DIV_2(x) ((x)-(x)%2)

extern "C" Coord get_upper_coord(const t_params * params)
{
    Coord res;
    int top = (params->n - params->m)/2;
    top = DIV_2(top);
    for (int d = 0; d < DIMS; ++d)
    {
        res.SetCoord(d, top);
    }
    return res;
}

extern "C" Coord get_lower_coord(const t_params * params)
{
    Coord res;
    int low = (params->n + params->m)/2;
    low = DIV_2(low);
    for (int d = 0; d < DIMS; ++d)
    {
        res.SetCoord(d, low);
    }
    return res;
}

bool is_internal(const t_params * params, const Coord & c)
{
	Coord up, down;
	up = get_upper_coord(params);
	down = get_lower_coord(params);
	bool res = true;
	for (int d = 0; d < DIMS; ++d)
	{
		if (!(up.GetCoord(d) <= c.GetCoord(d) && c.GetCoord(d) < down.GetCoord(d)))
		{
			res = false;
			break;
		}
	}
	return res;
}

extern "C" int get_params(t_params * params, char * filename)
{
    ini_fd_t ini;
    ini = ini_open(filename, "r", ";");
    //int res = 0;

    if (ini == NULL) {
        fprintf(stderr, "cannot parse file: %s\n", filename);
        return 1;
    }
    
    //double notfound = NOT_FOUND_VAL;
    int ret = ini_locateHeading(ini, "Interactions");
    ret += ini_locateKey(ini, "D_A");
    ret += ini_readDouble(ini, &params->D_A);
    ret += ini_locateKey(ini, "D_D");
    ret += ini_readDouble(ini, &params->D_D);
    ret += ini_locateKey(ini, "D_E");
    ret += ini_readDouble(ini, &params->D_E);
    ret += ini_locateKey(ini, "A_A");
    ret += ini_readDouble(ini, &params->A_A);
    ret += ini_locateKey(ini, "A_E");
    ret += ini_readDouble(ini, &params->A_E);
    ret += ini_locateKey(ini, "E_E");
    ret += ini_readDouble(ini, &params->E_E);
    ret += ini_locateKey(ini, "NeightboorsCnt");
    ret += ini_readInt(ini, &params->neigh_cnt);
    
    ret += ini_locateHeading(ini, "Computations");
    ret += ini_locateKey(ini, "m");
    ret += ini_readInt(ini, &params->m);
    ret += ini_locateKey(ini, "n");
    ret += ini_readInt(ini, &params->n);
    ret += ini_locateKey(ini, "DrugCnt");
    ret += ini_readInt(ini, &params->drug_cnt);
    ret += ini_locateKey(ini, "IterCnt");
    ret += ini_readInt(ini, &params->max_iter);
    
    if ( ini_locateKey(ini, "Device") != 0)
    {
    	params->device = 0;
    }
    else
    {
    	ret += ini_readInt(ini, &params->device);
    }
    
    params->struct_filename = (char *) malloc(sizeof(char)*256);
    ret += ini_locateKey(ini, "StructFile");
    ini_readString(ini, params->struct_filename, 255);
    
    char * filling = (char *) malloc(sizeof(char)*5);
    ret += ini_locateKey(ini, "Filling");
    ini_readString(ini, filling, 4);
    if (strcmp(filling, "A") == 0)
    {
        params->filling = FILL_ALL;
    }
    else if (strcmp(filling, "I") == 0)
    {
        params->filling = FILL_INTERNAL;
    }
    else if (strcmp(filling, "E") == 0)
    {
        params->filling = FILL_EXTERNAL;
    }
    else
    {
        printf("Internal filling will be used\n");
        params->filling = FILL_INTERNAL;
    }
    free(filling);
    
    params->neg_filename = NULL;
    if ( ini_locateKey(ini, "NegativeFile") == 0)
    {
    	params->neg_filename = (char *) malloc(sizeof(char)*256);
    	ini_readString(ini, params->neg_filename, 255);
    }
    
    ret += ini_locateHeading(ini, "Statistics");
    ret += ini_locateKey(ini, "SaveStatEvery");
    ret += ini_readInt(ini, &params->count_every);
    if (ini_locateKey(ini, "SaveBmp") != -1)
    {
    	ret += ini_readBool(ini, &params->save_bmp);
    }
    
    params->print_to = (char *) malloc(sizeof(char)*256);
    ret += ini_locateKey(ini, "LogFile");
    ini_readString(ini, params->print_to, 255);
	
	if (ini_locateHeading(ini, "Dump"))
	{
		params->load_dmp = (int)false;
		if (ini_locateKey(ini, "LoadDump") == 0)
		{
			ret += ini_readBool(ini, &params->load_dmp);
		}
		if (params->load_dmp)
		{
			ret += ini_locateKey(ini, "DumpFile");
			params->dmp_file = (char *) malloc(sizeof(char)*256);
			ini_readString(ini, params->dmp_file, 255);
		}
	}
    
    ini_close(ini);
    if (check_params(params, ret) == 0)
    {
        printf("Ini readed\n");
        return 0;
    }
    fprintf(stderr, "Bad params file\n");
    return 1;
}

int check_params(t_params * params, int ret)
{
    if (ret != 0)
    {
        return 1;
    }
    double notfound = NOT_FOUND_VAL;
    if (params->D_A == notfound)
    {
        fprintf(stderr, "Bad D_A\n");
        return 1;
    }
    if (params->D_D == notfound)
    {
        fprintf(stderr, "Bad D_D\n");
        return 1;
    }
    if (params->D_E == notfound)
    {
        fprintf(stderr, "Bad D_E\n");
        return 1;
    }
    if (params->A_A == notfound)
    {
        fprintf(stderr, "Bad A_A\n");
        return 1;
    }
    if (params->A_E == notfound)
    {
        fprintf(stderr, "Bad A_E\n");
        return 1;
    }
    if (params->E_E == notfound)
    {
        fprintf(stderr, "Bad E_E\n");
        return 1;
    }
    
    if (params->n <= params->m)
    {
        fprintf(stderr, "n <= m!\n");
        return 1;
    }
    if (params->drug_cnt == 0)
    {
        fprintf(stderr, "No drug - no iterations needed\n");
        return 1;
    }
    if (params->max_iter == 0)
    {
        fprintf(stderr, "IterCnt == 0. No iterations needed\n");
        return 1;
    }
    return 0;
}

int fgeti(FILE *stream, int notfound);

extern "C" int fill_with_structure(t_params * params, CellsField * fld)
{
    FILE * pStruct = fopen(params->struct_filename, "r");
    if (pStruct == NULL)
    {
        fprintf(stderr, "Error opening structure file\n");
        return 1;
    }
    
    int notfound = -1;
    int res = 0;
    
    while (!feof(pStruct))
    {
        Coord curr_coord;
        for (int d = 0; d < DIMS; ++d)
        {
            int curr_d = fgeti(pStruct, notfound);
            if (curr_d == notfound)
            {
            	if (d == 0)
            	{
            		res = 0;
            		goto exit;
            	}
            	
                fprintf(stderr, "Wrong structure file\n");
                res = 1;
                goto exit;
            }
            if (curr_d > params->m)
            {
                fprintf(stderr, "Wrong m!\n");
                res = 1;
                goto exit;
            }
            curr_coord.SetCoord(d, curr_d);
        }
        fld->SetElementVal(curr_coord + get_upper_coord(params), LABEL_AG);
    }

exit:
    fclose(pStruct);
    return res;
}

int fgeti(FILE *stream, int notfound)
{
    int res = notfound;
	int sign = 1;
    while (!feof(stream))
    {
        char c = fgetc(stream);
		if (c == '-')
		{
			sign = -1;
		}
        else if ('0' <= c && c <= '9')
        {
            if (res == notfound)
            {
                res = 0;
            }
            res *= 10;
            res += (int)(c-'0');
            continue;
        }
        else if (res == notfound)
        {
        	continue;
        }
        break;
    }
    return sign * res;
}

extern "C" CoordVec * load_struct (char * filename)
{
    ifstream fin(filename);
    CoordVec * res = new CoordVec();
    
    Coordinate x = 0, y = 0, z = 0;
    while (!fin.eof())
    {
        #if (DIMS == 2)
        fin >> x >> y;
        #else
        fin >> x >> y >> z;
        #endif
        res->push_back(Coord(x,y,z));
    }
    fin.close();
    return res;
}

void filter_negative(t_params * params, CoordVec * negative)
{
    const Coord top = get_upper_coord(params);
    const Coord low = get_lower_coord(params);
    Coord curr_coord;
    if (params->filling == FILL_INTERNAL)
    {
        int index = negative->size()-1;
        for (; index >= 0; --index)
        {
        	curr_coord = (*negative)[index];
            if (!is_internal(params, curr_coord))
            {
                negative->erase(negative->begin()+index);
            }
        }
    }
    else if (params->filling == FILL_EXTERNAL)
    {
        int index = negative->size()-1;
        for (; index >= 0; --index)
        {
        	curr_coord = (*negative)[index];
            if (is_internal(params, curr_coord))
            {
                negative->erase(negative->begin()+index);
            }
        }
    }
}


void shift_negative(t_params * params, CoordVec * negative)
{
    CoordVec::iterator it;
    Coord up = get_upper_coord(params);
    for(it = negative->begin(); it != negative->end(); ++it)
    {
        *it = *it + up;
    }
}

extern "C" int fill_with_drug(t_params * params, CellsField * fld)
{
    CoordVec * negative;
    Coord up, down;
    up = get_upper_coord(params);
    down = get_lower_coord(params);
    if (params->neg_filename != NULL)
    {
        negative = load_struct(params->neg_filename);
        shift_negative (params, negative);
        Coord curr_coord;
        Coordinate z = 0;
        for (Coordinate x=0; x < params->n; ++x)
        {
            for (Coordinate y=0; y < params->n; ++y)
            {
                #if (DIMS ==3)
                for (z = 0; z < params->n; ++z)
                {
                #endif
                curr_coord = Coord(x,y,z);
                
                if (!is_internal(params, curr_coord))
                {
                    negative->push_back(curr_coord);
                }
                #if (DIMS ==3)
                }
                #endif
            }
        }
    }
    else
    {
        Coordinate z = 0;
        negative = new CoordVec();
        for (Coordinate x=0; x < params->n; ++x)
        {
            for (Coordinate y=0; y < params->n; ++y)
            {
                #if (DIMS ==3)
                for (z = 0; z < params->n; ++z)
                #endif
                if (fld->GetElementVal(Coord(x,y,z)) != LABEL_AG)
                {
                    negative->push_back(Coord(x,y,z));
                }
            }
        }
    }
    
    filter_negative(params, negative);
    
    if ((unsigned)params->drug_cnt > negative->size())
    {
        fprintf(stderr, "There is not enougth space to drug\n");
        delete negative;
        return 1;
    }
    
    random_shuffle(negative->begin(), negative->end());
    CoordVec::iterator it;
    int index;
    for (index = 0, it = negative->begin(); index < params->drug_cnt; ++index, ++it)
    {
        fld->SetElementVal(*it, LABEL_DRUG);
    }
    delete negative;
    return 0;
}

extern "C" bool file_exists(const char * filename)
{
    if (FILE * file = fopen(filename, "r"))
    {
        fclose(file);
        return true;
    }
    return false;
}

extern "C" void get_rotation_maps(CellsField * fld, RotationType * even, RotationType * odd)
{
    Coord sz = fld->GetSize();
    for (int x = 0; x < sz.GetCoord(0); x+=2)
    {
        for (int y = 0; y < sz.GetCoord(1); y+=2)
        {
            bool curr_rotablility_even = true;
            bool curr_rotablility_odd = true;
            for (int dx = 0; dx < 2; ++dx)
            {
                for (int dy = 0; dy < 2; ++dy)
                {
                	ElementType lbl_even = fld->GetElementVal(Coord(x+dx, y+dy));
                    if (lbl_even == LABEL_AG)
                    {
                        curr_rotablility_even = false;
                    }
                    ElementType lbl_odd = fld->GetElementVal(Coord((x+dx+1)%sz.GetCoord(0),(y+dy+1)%sz.GetCoord(1)));
                    if (lbl_odd == LABEL_AG)
                    {
                        curr_rotablility_odd = false;
                    }
                }
            }
            
            even[x/2+y*sz.GetCoord(0)/4] = (RotationType)curr_rotablility_even;
            odd[x/2+y*sz.GetCoord(0)/4] = (RotationType)curr_rotablility_odd;
        }
    }
}


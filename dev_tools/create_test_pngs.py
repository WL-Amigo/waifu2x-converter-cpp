#!/usr/bin/env python3

# Creates every possible png variant based on input.png (make sure to cut some holes into input.png to see the alpha!)
# Generated html contains many and (possibly) big images, seems to load way better+faster in firefox than chrome (for me anyway).

# Only works/tested on Linux/WSL

import os

INPUT_FILE = "input.png"


TEST_HTML = '''<!DOCTYPE html>
<html>
\t<head>
\t\t<meta charset="utf-8">
\t\t<style>
\t\tbody
\t\t{{
\t\tbackground: rgba(255,0,0,1);
\t\tbackground: linear-gradient(to bottom, rgba(255,0,0,1) 0%, rgba(239,1,124,1) 69%, rgba(239,1,124,1) 100%);
\t\t}}
\t\tth {{ font-size: 15pt; background-color: white; }}
\t\t</style>
\t</head>
\t<body>
{table}
\t</body>
</html>'''

names = {
	"grayscale" : {
		"png_type": 0,
		"bits": {
			1: 1,
			2: 2,
			4: 4,
			8: 8,
			16: 16,
		}
	},
	"grayscale_alpha" :	{
		"png_type": 4,
		"bits": {
			8: 16,
			16: 32,
		}
	},
	"indexed" :	{
		"png_type": 3,
		"bits": {
			1: 1,
			2: 2,
			4: 4,
			8: 8,
		}
	},
	"indexed_alpha" : { # not sure if this is valid.
		"png_type": 3,
		"bits": {
			1: 4,
			2: 8,
			4: 16,
			# 8: 32, # convert-im6.q16: Valid palette required for paletted images `indexed_alpha_8-16bit_type_3.png' @ error/png.c/MagickPNGErrorHandler/1628.
		}
	},
	"truecolor" : {
		"png_type": 2,
		"bits": {
			8: 24,
			16: 48,
		}
	},
	"truecolor_alpha" : {
		"png_type": 6,
		"bits": {
			8: 32,
			16: 64,
		}
	}
}

total_images = sum([len(names[name]["bits"]) for name in names])
created_images = []

for name in names:
	for bit_depth, bit_depth2 in names[name]["bits"].items():
	
		imagick_cmd = "convert {input_alpha_png}{alpha} -depth {depth} {name_extra}"
		
		png_type = names[name]["png_type"]
		alpha = 1 if name.endswith("_alpha") else 0
		name_extra = ""
		pipe_cmd = None
		if name.startswith("grayscale_"):
			name_extra = " -set colorspace Gray "
		elif name.startswith("indexed_"):
			# name_extra = " -trim +repage -colors 256 -type palette"
			# name_extra = " -trim -colors 16 -type palette"
			name_extra = " -trim -colors 256 -type palettematte GIF:- "
			pipe_cmd = 'convert - -depth {depth} "{output_image}"'
			alpha = 2
		print("Creating {name} PNG#{current_image}/{total_images} @ {depth},{depth2}bits, Alpha: {alpha}".format(
			depth=bit_depth,
			depth2=bit_depth2,
			name=name,
			alpha="Yes" if alpha == 1 else "no",
			current_image=len(created_images),
			total_images=total_images
		))
		output_image = name + "_" + str(bit_depth) + "-" + str(bit_depth2)+ "bit" + "_type_" + str(png_type) + ".png"
		_alpha = " -alpha off"
		if alpha == 1:
			_alpha = " -alpha on"
		elif alpha == 2:
			_alpha = ""	

		if pipe_cmd is None:
			imagick_cmd += "{output_image}"
		
		if pipe_cmd is not None:
			imagick_cmd = imagick_cmd + " | " + pipe_cmd
		
		cmd = imagick_cmd.format(
			input_alpha_png=INPUT_FILE,
			alpha=_alpha,
			depth=bit_depth,
			name_extra=name_extra,
			output_image=output_image
		)
		
		print(cmd)
		os.system(cmd)
		created_images.append(output_image)

column_width = 5
remainder_columns = total_images % column_width
screen_width = 1920
max_width = (screen_width/column_width)-14
curr_img = 0

table = '\t\t<table>\n'
for row in range(0,int(total_images / column_width)):
	header_trs = []
	img_trs = []
	for c in range(0, column_width):
		img = created_images[curr_img]
		header_trs.append('\t\t\t\t<th>' + img + ("<br><b>MUST HAVE ALPHA!</b>" if "_alpha" in img else "") + '</th>\n')
		img_trs.append('\t\t\t\t<td><img src="' + img + '" width="'+str(max_width)+'"></td>\n')
		curr_img+=1

	table += '\t\t\t<tr>\n'
	for td in header_trs:
		table += td 
	table += '\t\t\t</tr\n>'
	table += '\t\t\t<tr>\n'
	for td in img_trs:
		table += td 
	table += '\t\t\t</tr>\n'

if remainder_columns > 0:
	header_trs = []
	img_trs = []
	for c in range(0, remainder_columns):
		img = created_images[curr_img]
		header_trs.append('\t\t\t\t<th>' + img + ("<br><b>MUST HAVE ALPHA!</b>" if "_alpha" in img else "") + '</th>\n')
		img_trs.append('\t\t\t\t<td><img src="' + img + '" width="'+str(max_width)+'"></td>\n')
		curr_img+=1
	table += '\t\t\t<tr>\n'
	for td in header_trs:
		table += td 
	table += '\t\t\t</tr\n>'
	table += '\t\t\t<tr>\n'
	for td in img_trs:
		table += td 
	table += '\t\t\t</tr>\n'

table += '\t\t</table>'

table2 = table.replace(".png", "_[NS-L1][x2.000000].png")


with open("test.html", "w", encoding="utf-8") as f:
	f.write(TEST_HTML.format(table=table))
	
with open("test_scaled.html", "w", encoding="utf-8") as f:
	f.write(TEST_HTML.format(table=table2))
	
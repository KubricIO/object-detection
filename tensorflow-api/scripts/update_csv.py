def recalculate_vertices(width, filename, new_filename):
    new_lines = []
    with open(filename, 'r') as csv_file:
        line = csv_file.readline()
        line_list = line.split(',')
        if line_list[1] > width:
            factor = (width/line_list[1])
            new_xmin = str(factor*int(line_list[4]))
            new_ymin = str(factor*int(line_list[5]))
            new_xmax = str(factor*int(line_list[6]))
            new_ymax = str(factor*int(line_list[7]))
            new_line = ''.join([line_list[0], line_list[1], line_list[2], line_list[3], new_xmin, new_ymin, new_xmax, new_ymax])
        else:
            new_line = line
        new_lines.append(new_line)

    with open(new_filename, 'w') as csv_file:
        for line in new_lines:
            csv_file.write(line+"\n")
import os, argparse, ast , numpy as np, cv2

ap = argparse.ArgumentParser()

ap.add_argument( '-a' , '--input_ann_file' , required = True , type = str , help = 'path to input annotations.txt file' )
ap.add_argument( '-f' , '--frames_path' , required = True , type = str , help = 'path to frames' )
ap.add_argument( '-o' , '--output_path' , required = True , type = str , help = 'path to save the output files' )

args = vars(ap.parse_args())

print(args)

# Setting the input file to read
ann_file_to_read = args['input_ann_file']
frames_path = args['frames_path']

# path to save the plotted images  
output_path = args['output_path']

if not os.path.exists(output_path):
    os.makedirs(output_path)


# Reading the annotations.txt file
with open(ann_file_to_read, 'r') as af:
    annotations = af.readlines()
    
af.close()

annotations = [line.strip() for line in annotations] # TO remove the '\n' added at the end of line
    
# Setting up the txt details
    
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 2
fontColor              = (0,255,0)
lineType               = 2

count = 0

for ann in sorted(annotations):
    
    ann = ann.split(" ")
    
    #Setting the input and output file names
    
    frame_no = str(ann[0].split("/")[-1])

    print("\nReading ...............",frame_no) 
    
    read_file = frames_path+frame_no
    write_file = output_path+frame_no
    
    img = cv2.imread(read_file)
    
    if img.size != 0:
        
        count += 1

    ann = [ast.literal_eval(x) for x in ann[2:]]
    ann = np.array(ann).reshape(-1,5)
    
    for bb in ann:
    
        # bb = bb.split(",")
        
        # bb = [ast.literal_eval(b) for b in bb]
        
        cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[0] + bb[2]), int(bb[1] + bb[3])) , (0,255,255), 2 )
        
    cv2.imwrite(write_file,img)
    # cv2.waitKey(0)
    

    
print("\nNo of frames labelled in annotations file :", len(annotations))
print("\nNo of labels plotted on frames : ", count)
print("\n------Task Completed---------------\n")

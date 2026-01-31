# sort_cvlTrain_by_writer.py
input_path = "/cluster/datastore/aniketag/newHTR/icpr/HTR-best-practices/data/cvlTest.txt"
output_path = "/cluster/datastore/aniketag/newWordStylist/wordStylist2/WordStylist/gt/cvlTest_sorted.txt"

# Read and sort lines
with open(input_path, 'r') as f:
    lines = f.readlines()

# Remove any empty lines and strip whitespaces
lines = [line.strip() for line in lines if line.strip()]

# Sort lines using WriterID (first element before comma)
sorted_lines = sorted(lines, key=lambda x: x.split(',')[0])

# Write sorted content back
with open(output_path, 'w') as f:
    for line in sorted_lines:
        f.write(line + '\n')

print(f"✅ Sorting complete! Sorted file saved to: {output_path}")

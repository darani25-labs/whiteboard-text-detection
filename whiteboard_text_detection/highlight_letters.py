import cv2

# Load image
img_path = "image_file/whiteboard image.jpg"
image = cv2.imread(img_path)

# Check if image is loaded
if image is None:
    print("❌ Image not found! Check path and file name.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Remove small noise using morphological closing
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find contours (letter boundaries)
contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes around letters
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if 10 < w < 200 and 10 < h < 200:  # filter to ignore small dots or big regions
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Save and display the annotated image
cv2.imshow("Highlighted Letters", image)
cv2.imwrite("highlighted_whiteboard_letters.jpg", image)

print("✅ Letters highlighted successfully! Saved as 'highlighted_whiteboard_letters.jpg'.")

cv2.waitKey(0)
cv2.destroyAllWindows()

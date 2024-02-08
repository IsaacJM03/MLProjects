from PIL import Image
import os

def convert_to_grayscale(input_dir, output_dir):
  """
  Converts all images in a directory and its subdirectories to grayscale and saves them in another directory.

  Args:
    input_dir: Path to the directory containing the images to convert.
    output_dir: Path to the directory where the grayscale images will be saved.

  Raises:
    ValueError: If either input_dir or output_dir is not a valid directory.
  """

  for root, dirs, files in os.walk(input_dir):
    for filename in files:
      if not filename.endswith((".jpg", ".png", ".jpeg")):
        continue

      player_name = os.path.basename(root)
      input_path = os.path.join(root, filename)
      output_path = os.path.join(output_dir, player_name, filename)

      # Create output directory for each player if it doesn't exist
      os.makedirs(os.path.dirname(output_path), exist_ok=True)

      try:
        img = Image.open(input_path)
        grayscale_img = img.convert("L")
        grayscale_img.save(output_path)
        print(f"Converted {filename} for player {player_name} to grayscale")
      except Exception as e:
        print(f"Error converting {filename} for player {player_name}: {e}")

if __name__ == "__main__":
  input_dir = "/home/isaac-flt/Projects/ML4D/MLProjects/footballers model/Images/Images/Players/ALL-PLAYERS"  # Replace with your actual input directory
  output_dir = "/home/isaac-flt/Projects/ML4D/MLProjects/footballers model/Images/Images/Players/grayscale"  # Replace with your desired output directory

  try:
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    convert_to_grayscale(input_dir, output_dir)
  except ValueError as e:
    print(e)
  print("Done! Grayscale images saved in", output_dir)

from PIL import Image
import os
import rembg
import io

def remove_background_and_convert_to_grayscale(input_dir, output_dir):
    """
    Removes background from images and converts them to grayscale.

    Args:
        input_dir (str): Path to the directory containing the images to process.
        output_dir (str): Path to the directory where processed images will be saved.
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
                with open(input_path, "rb") as input_file:
                    input_data = input_file.read()

                # Remove background using Rembg
                output_data = rembg.remove(input_data)

                # Convert to PIL Image
                img = Image.open(io.BytesIO(output_data))
                
                # Convert to grayscale
                grayscale_img = img.convert("L")

                # Save the processed image
                grayscale_img.save(output_path)
                print(f"Processed {filename} for player {player_name}")
            except Exception as e:
                print(f"Error processing {filename} for player {player_name}: {e}")

if __name__ == "__main__":
    input_dir = ""  # Replace with your actual input directory
    output_dir = ""  # Replace with your desired output directory

    try:
        os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
        remove_background_and_convert_to_grayscale(input_dir, output_dir)
    except ValueError as e:
        print(e)

    print("Done! Processed images saved in", output_dir)

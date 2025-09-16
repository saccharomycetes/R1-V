from PIL import Image, ImageDraw
import random
import json
import os
import argparse

def generate_connected_squares(grid_size=20, num_squares=15):
    """
    Generate randomly connected square blocks.
    
    Args:
        grid_size: Size of the grid for placing squares
        num_squares: Number of squares to generate
    """
    # Create a grid to track which positions have squares
    grid = [[False for _ in range(grid_size)] for _ in range(grid_size)]
    squares = []
    
    # Start from a random position near the center
    start_x = grid_size // 2 + random.randint(-2, 2)
    start_y = grid_size // 2 + random.randint(-2, 2)
    
    # Add the first square
    grid[start_y][start_x] = True
    squares.append((start_x, start_y))
    
    # Keep track of positions that can be expanded from
    frontier = [(start_x, start_y)]
    
    # Directions: up, down, left, right
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    
    # Generate connected squares
    while len(squares) < num_squares and frontier:
        # Pick a random square from the frontier
        base_idx = random.randint(0, len(frontier) - 1)
        base_x, base_y = frontier[base_idx]
        
        # Find valid adjacent positions
        valid_positions = []
        for dx, dy in directions:
            new_x, new_y = base_x + dx, base_y + dy
            if (0 <= new_x < grid_size and 
                0 <= new_y < grid_size and 
                not grid[new_y][new_x]):
                valid_positions.append((new_x, new_y))
        
        if valid_positions:
            # Choose a random valid position
            new_x, new_y = random.choice(valid_positions)
            grid[new_y][new_x] = True
            squares.append((new_x, new_y))
            frontier.append((new_x, new_y))
        else:
            # Remove this position from frontier if no valid neighbors
            frontier.pop(base_idx)
    
    return squares

def draw_connected_squares(squares, square_size=30, padding=50):
    """
    Draw the connected squares with black outlines and light grey fill.
    """
    # Find bounding box
    if not squares:
        return None
    
    xs = [s[0] for s in squares]
    ys = [s[1] for s in squares]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Calculate image size
    width = (max_x - min_x + 1) * square_size + 2 * padding
    height = (max_y - min_y + 1) * square_size + 2 * padding
    
    # Create image with white background
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Define colors
    light_grey = (192, 192, 192)  # Light grey fill
    black = (0, 0, 0)  # Black outline
    
    # Draw each square
    for x, y in squares:
        # Calculate pixel position (adjust for minimum values)
        pixel_x = (x - min_x) * square_size + padding
        pixel_y = (y - min_y) * square_size + padding
        
        # Draw rectangle with light grey fill and black outline
        draw.rectangle(
            [pixel_x, pixel_y, pixel_x + square_size, pixel_y + square_size],
            fill=light_grey,
            outline=black,
            width=2
        )
    
    return img

def generate_random_shape(max_squares=25):
    """
    Generate a completely random connected shape.
    """
    num_squares = random.randint(1, max_squares)
    grid_size = max(10, num_squares * 2)  # Ensure enough space
    
    squares = generate_connected_squares(grid_size, num_squares)
    return squares

def generate_dataset(num_images=10, max_squares=25, square_size=40, output_dir='./data'):
    """
    Generate a dataset of connected square images with metadata.
    
    Args:
        num_images: Number of images to generate
        max_squares: Maximum number of squares per image (random from 1 to max)
        square_size: Size of each square in pixels
        output_dir: Directory to save images and metadata
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Store metadata for all images
    metadata = []
    
    # print(f"Generating {num_images} images with 1-{max_squares} blocks each...")
    # print(f"Output directory: {output_dir}")
    
    for i in range(num_images):
        # print(f"\nGenerating image {i+1}/{num_images}...")
        
        # Generate random connected squares
        squares = generate_random_shape(max_squares=max_squares)
        num_blocks = len(squares)
        # print(f"Generated {num_blocks} connected squares")
        
        # Draw the squares
        img = draw_connected_squares(squares, square_size=square_size)
        
        if img:
            # Save the image
            filename = f'grid_image_{i+1:04d}.png'
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
            # print(f"Saved to {filepath}")
            
            # Store metadata
            metadata.append({
                'image_id': i + 1,
                'filename': filename,
                'num_blocks': num_blocks
            })
    
    # Save metadata to JSON file
    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump({
            'dataset_info': {
                'total_images': num_images,
                'min_blocks_per_image': 1,
                'max_blocks_per_image': max_squares,
                'square_size_pixels': square_size
            },
            'images': metadata
        }, f, indent=2)
    
    print(f"\nMetadata saved to {metadata_file}")
    print(f"\nDataset generation complete! Generated {num_images} images.")
    
    # Print summary statistics
    total_blocks = sum(img['num_blocks'] for img in metadata)
    avg_blocks = total_blocks / num_images if num_images > 0 else 0
    print(f"Average blocks per image: {avg_blocks:.2f}")
    print(f"Total blocks generated: {total_blocks}")

# Generate multiple random shapes
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate dataset of connected square grids')
    parser.add_argument('--num_images', type=int, default=10000, help='Number of images to generate (default: 1000)')
    parser.add_argument('--max_squares', type=int, default=15, help='Maximum number of squares per image, randomized from 1 to max (default: 15)')
    parser.add_argument('--square_size', type=int, default=40, help='Size of each square in pixels (default: 40)')
    parser.add_argument('--output_dir', type=str, default='./data', help='Output directory for images and metadata (default: ./data)')
    
    args = parser.parse_args()
    
    # Generate the dataset
    generate_dataset(
        num_images=args.num_images,
        max_squares=args.max_squares,
        square_size=args.square_size,
        output_dir=args.output_dir
    )
# POLLICLIENT-GUI

POLLICLIENT-GUI is a PyQt5-based application that allows users to generate images from prompts and descriptions using external APIs. This tool provides a convenient interface for handling image descriptions, prompt enhancement, and thumbnail viewing for generated images. It also allows for the management of treatments and prompt history.

## Features

- **Image to Prompt**: Extract detailed descriptions from images.
- **Prompt Enhancement**: Enhance existing prompts using external models.
- **Thumbnail Viewer**: Display and manage thumbnails of generated images.
- **Custom Treatments**: Define and apply custom pre and post prompts to enhance descriptions.
- **Prompt History**: Save and reuse previously used prompts.

## Installation

1. Ensure you have Python 3.x installed.
2. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python POLLICLIENT-GUI.py
   ```

## Usage

- **Image to Prompt**: Select an image to generate a detailed description using external APIs.
- **Enhance Prompt**: Input a prompt and enhance it using the selected model.
- **Thumbnail Viewer**: View and manage previously generated images.
- **Treatment Management**: Create, edit, or delete treatments to modify how prompts are enhanced.
  
## Dependencies

- PyQt5
- PIL (Pillow)
- aiohttp
- requests
- numpy
- asyncio
- json
- base64

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

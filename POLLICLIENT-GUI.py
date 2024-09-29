from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QComboBox, QPushButton, QLineEdit, QTextEdit, QVBoxLayout, QHBoxLayout, QGridLayout, QWidget, QMenuBar, QMenu, QDialog, QInputDialog, QScrollArea, QSizePolicy, QCheckBox, QRadioButton, QButtonGroup, QFileDialog
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import QTimer, QSize, QPoint, QMetaObject, Qt, Q_ARG, QThread, QObject, pyqtSignal, pyqtSlot, QBuffer
from PIL import Image
import numpy as np
import asyncio
import aiohttp
import os
import datetime
import random
import urllib.parse
import threading
import io
import json
import signal
import sys
import requests
import re
import base64 

def get_ai_instruction(task_type, original_content=""):
    base_instruction = """
    Provide a detailed description in a single paragraph of plain English. 
    Include specific details such as genders, names, colors, positioning, styles, and any other relevant information. 
    Use underscores to connect words that should be treated as a single concept (e.g., red_car, tall_man). 
    Use commas to separate different elements or ideas in the description. 
    Ensure the description is comprehensive enough for accurate recreation or enhancement.
    """
    
    task_specific_instructions = {
        "image_to_prompt": "Analyze this image and recreate the prompt likely used to generate it. " + base_instruction,
        "enhance_prompt": f"Enhance the following prompt for text-to-image generation. {base_instruction} The original prompt is: {{original_content}}",
        "describe_image": "Describe this image fully for recreation. " + base_instruction
    }
    
    return task_specific_instructions[task_type].format(original_content=original_content)
    
signal.signal(signal.SIGINT, signal.SIG_DFL)

class ThumbnailUpdateSignal(QObject):
    signal = pyqtSignal(str)

class ImageToPromptWorker(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        try:
            print(f"Attempting to process image: {self.image_path}")
            
            # Read and encode the image
            with open(self.image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Construct the API request
            api_url = "https://text.pollinations.ai/"
            headers = {
                "Content-Type": "application/json"
            }
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": get_ai_instruction("image_to_prompt")
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "model": "openai",
                "seed": -1,
                "jsonMode": False
            }

            # Send the request
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            # The response should now be plain text
            description = response.text.strip()
            print(f"Generated description: {description}")
            self.finished.emit(description)

        except Exception as e:
            print(f"Error in image_to_prompt: {str(e)}")
            self.error.emit(str(e))

class PromptEnhancer(QObject):
    finished = pyqtSignal(str)

    def enhance_prompt(self, prompt, model):
        enhanced_prompt = asyncio.run(self.enhance_prompt_with_api(prompt, model))
        return enhanced_prompt

    async def enhance_prompt_with_api(self, prompt, model):
        formatted_request = get_ai_instruction("enhance_prompt", original_content=prompt)
        
        base_url = "https://text.pollinations.ai/"
        url = f"{base_url}{urllib.parse.quote(formatted_request)}&model={urllib.parse.quote(model)}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        response_text = await response.text()
                        print(f"Raw response from API: {response_text}")  # Debugging line
                        return response_text.strip()
                    else:
                        print(f"Error enhancing prompt with API: {response.status}")
                        return prompt
        except Exception as e:
            print(f"Error in enhance_prompt_with_api: {e}")
            return prompt
            
class AsyncThread(QObject):
    finished = pyqtSignal(object)

    def __init__(self, url):
        super().__init__()
        self.url = url

    def run(self):
        try:
            result = asyncio.run(self.fetch_models_async())
            self.finished.emit(result)
        except Exception as e:
            print(f"Error fetching models: {e}")
            self.finished.emit([])

    async def fetch_models_async(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(self.url) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"Fetched data from {self.url}: {data}")  # Debug print
                    return data
                else:
                    print(f"Error fetching models: HTTP {response.status}")
                    return []

class ThumbnailViewer(QDialog):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle("Thumbnail Viewer")
        self.setMinimumSize(900, 600)  # Increased width to accommodate more columns
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)

        # Main layout
        main_layout = QVBoxLayout(self)

        # Scroll area for thumbnails
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.thumbnail_widget = QWidget()
        self.thumbnail_layout = QGridLayout(self.thumbnail_widget)
        self.thumbnail_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.thumbnail_layout.setSpacing(5)  # Reduce spacing between thumbnails
        self.scroll_area.setWidget(self.thumbnail_widget)
        main_layout.addWidget(self.scroll_area)

        self.thumbnails = []
        self.max_columns = 8  # Increased from 6 to 8
        self.max_thumbnails = 80  # Increased from 50 to 80

    def load_thumbnails(self, directory):
        self.clear_thumbnail_layout()
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort(key=lambda x: os.path.getctime(os.path.join(directory, x)), reverse=True)
        
        for index, filename in enumerate(image_files[:self.max_thumbnails]):
            image_path = os.path.join(directory, filename)
            QTimer.singleShot(0, lambda p=image_path, i=index: self.update_thumbnail(p, i))

        self.thumbnail_widget.adjustSize()

    def update_thumbnail(self, image_path, index):
        try:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                row = index // self.max_columns
                col = index % self.max_columns
                
                label = QLabel()
                label.setPixmap(pixmap)
                label.setFixedSize(100, 100)
                label.mousePressEvent = lambda event, path=image_path: self.thumbnail_clicked(event, path)
                self.thumbnail_layout.addWidget(label, row, col)
            else:
                print(f"Failed to load image: {image_path}")
        except Exception as e:
            print(f"Error updating thumbnail: {e}")

    def thumbnail_clicked(self, event, image_path):
        if event.button() == Qt.LeftButton:
            self.main_window.display_image(QPixmap(image_path))
        elif event.button() == Qt.RightButton:
            self.show_context_menu(event.globalPos(), image_path)

    def show_context_menu(self, position, image_path):
        context_menu = QMenu(self)
        copy_action = context_menu.addAction("Copy Image")
        delete_action = context_menu.addAction("Delete Image")
        
        action = context_menu.exec_(position)
        if action == copy_action:
            self.copy_image_to_clipboard(image_path)
        elif action == delete_action:
            self.delete_thumbnail(image_path)

    def copy_image_to_clipboard(self, image_path):
        image = QImage(image_path)
        clipboard = QApplication.clipboard()
        clipboard.setImage(image)
        self.main_window.status_bar.setText("Image copied to clipboard")

    def delete_thumbnail(self, image_path):
        try:
            os.remove(image_path)
            self.main_window.status_bar.setText(f"Image deleted: {image_path}")
            self.load_thumbnails(self.main_window.save_path)  # Reload thumbnails
        except OSError as e:
            self.main_window.status_bar.setText(f"Error deleting image: {e}")

    def clear_thumbnail_layout(self):
        while self.thumbnail_layout.count():
            item = self.thumbnail_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def sort_by_date(self):
        self.load_thumbnails(self.main_window.save_path)  # Default sort is by date

    def sort_by_name(self):
        self.clear_thumbnail_layout()
        if os.path.exists(self.main_window.save_path):
            files = sorted(
                [f for f in os.listdir(self.main_window.save_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                key=lambda x: x.lower()
            )
            for index, filename in enumerate(files[:self.max_thumbnails]):
                image_path = os.path.join(self.main_window.save_path, filename)
                self.update_thumbnail(image_path, index)
        self.thumbnail_widget.adjustSize()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_start_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.dragging:
            self.move(event.globalPos() - self.drag_start_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        self.dragging = False
        event.accept()

class TreatmentDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, initial_pre=None, initial_post=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Treatment")
        self.setGeometry(200, 200, 400, 300)

        layout = QVBoxLayout()

        self.pre_prompt = QTextEdit()
        self.pre_prompt.setPlaceholderText("Pre-prompt")
        if initial_pre:
            self.pre_prompt.setText(initial_pre)
        layout.addWidget(self.pre_prompt)

        self.post_prompt = QTextEdit()
        self.post_prompt.setPlaceholderText("Post-prompt")
        if initial_post:
            self.post_prompt.setText(initial_post)
        layout.addWidget(self.post_prompt)

        btn_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.accept)
        btn_layout.addWidget(save_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_button)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def get_results(self):
        return self.pre_prompt.toPlainText().strip(), self.post_prompt.toPlainText().strip()

class ImageGeneratorApp(QMainWindow):
    update_prompt_signal = pyqtSignal(str)
    thumbnail_update_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("POLLICLIENT Image Generator")
        self.setFixedSize(540, 840)
        self.async_thread = None
        
        self.on_top = False
        self.treatments_path = ''
        self.prompt_history_file = 'prompt_history.json'
        self.save_path = os.path.abspath('./GENERATED')
        self.generating_image = False
        self.last_seed = None

        self.treatments = {}
        self.prompt_history = []
        self.image_models = []
        self.text_models = []
    
        # Initialize UI components
        main_layout = self.initUI()

        self.load_settings()

        self.treatments = self.sync_treatments()
        self.update_treatment_combobox()
        self.prompt_history = self.load_prompt_history()
        self.update_history_combobox()
        
        self.history_combobox.setCurrentIndex(-1)
        self.prompt_entry.clear()

        self.thumbnail_viewer = ThumbnailViewer(self)
        self.thumbnail_viewer.setParent(self)
        self.thumbnail_viewer.setWindowFlags(self.thumbnail_viewer.windowFlags() | Qt.Window)
        self.thumbnail_viewer.hide()

        self.thumbnail_update_signal.connect(self.update_thumbnail)

        self.load_existing_thumbnails()

        thumbnail_button = QPushButton("Show Thumbnails")
        thumbnail_button.clicked.connect(self.show_thumbnail_viewer)

        self.fetch_models()
        
        self.prompt_enhancer = PromptEnhancer()
        self.prompt_enhancer.finished.connect(self.update_prompt_entry)

        self.original_prompt = ""
        self.enhanced_prompt = ""
        self.rtist_enhanced = False
        self.update_prompt_signal.connect(self.update_prompt_entry)

    @pyqtSlot(str)
    def update_thumbnail(self, image_path):
        if self.thumbnail_viewer and self.thumbnail_viewer.isVisible():
            self.thumbnail_viewer.load_thumbnails(self.save_path)

    def image_to_prompt(self, image_path):
        try:
            print(f"Attempting to process image: {image_path}")
            
            # Read and encode the image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Construct the API request
            api_url = "https://text.pollinations.ai/"
            headers = {
                "Content-Type": "application/json"
            }
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": get_ai_instruction("describe_image")
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "model": "openai",
                "seed": -1,
                "jsonMode": False
            }

            # Send the request
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            # The response should now be plain text
            description = response.text.strip()
            print(f"Generated description: {description}")
            return description

        except Exception as e:
            print(f"Error in image_to_prompt: {str(e)}")
            return f"Error: {str(e)}"
        
    def select_and_display_image(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if image_path:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
                self.status_bar.setText(f"Image loaded: {image_path}")
                
                # Store the path of the selected image instead of saving a copy
                self.last_loaded_image_path = image_path
            else:
                self.status_bar.setText("Failed to load the selected image.")

    def on_image_to_prompt_click(self):
        if hasattr(self, 'last_loaded_image_path') and os.path.exists(self.last_loaded_image_path):
            self.status_bar.setText("Generating description from image...")
            
            # Create a QThread object
            self.thread = QThread()
            # Create a worker object
            self.worker = ImageToPromptWorker(self.last_loaded_image_path)
            # Move worker to the thread
            self.worker.moveToThread(self.thread)
            # Connect signals and slots
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.finished.connect(self.update_prompt_with_description)
            self.worker.error.connect(self.handle_image_to_prompt_error)
            # Start the thread
            self.thread.start()

        else:
            self.status_bar.setText("No image available to generate description.")

    def update_prompt_with_description(self, description):
        self.prompt_entry.setText(description)
        self.status_bar.setText("Description generated from the image.")

    def handle_image_to_prompt_error(self, error):
        self.status_bar.setText(f"Error generating description: {error}")

    def update_image_model_combobox(self):
        current_model = self.model_combobox.currentText()
        self.model_combobox.clear()
        if self.image_models:
            self.model_combobox.addItems(self.image_models)
            index = self.model_combobox.findText(current_model)
            if index >= 0:
                self.model_combobox.setCurrentIndex(index)
            else:
                self.model_combobox.setCurrentIndex(0)
        else:
            print("No image models available")
     
    def cleanup(self):
        if hasattr(self, 'image_thread') and self.image_thread.isRunning():
            self.image_thread.quit()
            self.image_thread.wait()
        if hasattr(self, 'text_thread') and self.text_thread.isRunning():
            self.text_thread.quit()
            self.text_thread.wait()
            
    def show_thumbnail_viewer(self):
        if self.thumbnail_viewer.isVisible():
            self.thumbnail_viewer.hide()
        else:
            self.thumbnail_viewer.show()
            self.thumbnail_viewer.raise_()
            self.thumbnail_viewer.activateWindow()
            self.thumbnail_viewer.load_thumbnails(self.save_path)
        
    def load_existing_thumbnails(self):
        if os.path.exists(self.save_path):
            self.thumbnail_viewer.load_thumbnails(self.save_path)

    def display_image(self, image):
        if image is None:
            print("No image to display")
            return

        try:
            if isinstance(image, QPixmap):
                pixmap = image
            elif isinstance(image, QImage):
                pixmap = QPixmap.fromImage(image)
            else:
                print(f"Unsupported image type: {type(image)}")
                return

            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            print("Image displayed on the label")

            # Update the thumbnail viewer
            self.thumbnail_viewer.load_thumbnails(self.save_path)
        except Exception as e:
            print(f"Failed to display image: {e}")
            import traceback
            traceback.print_exc()

    def download_and_display_image(self, url):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                image_data = response.content
                image = Image.open(io.BytesIO(image_data))
                save_path = self.save_image(image)
                qimage = self.pil_to_qimage(image)

                if qimage:
                    self.display_image(qimage)
            else:
                self.status_bar.setText(f"Error: Failed to retrieve image. Status code: {response.status_code}")
        except Exception as e:
            self.status_bar.setText(f"Error downloading image: {str(e)}")
            import traceback
            traceback.print_exc()

    def revert_to_original_prompt(self):
        if hasattr(self, 'original_prompt'):
            self.prompt_entry.setText(self.original_prompt)
            self.status_bar.setText("Reverted to original prompt")
        else:
            self.status_bar.setText("No original prompt to revert to")
        
    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)

        menubar = QMenuBar(self)
        options_menu = QMenu("Options", self)
        self.on_top_action = options_menu.addAction("On Top", self.toggle_on_top)
        self.on_top_action.setCheckable(True)
        self.on_top_action.setChecked(self.on_top)
        options_menu.addAction("Set Treatments Path", self.set_treatments_path)
        menubar.addMenu(options_menu)
        self.setMenuBar(menubar)

        treatment_layout = QHBoxLayout()
        treatment_layout.setContentsMargins(0, 0, 0, 0)
        treatment_label = QLabel("Treatment:")
        self.treatment_combobox = QComboBox()
        self.treatment_combobox.addItems(["None"])
        self.treatment_combobox.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        
        treatment_layout.addWidget(treatment_label)
        treatment_layout.addWidget(self.treatment_combobox)
        treatment_layout.addStretch(1)

        new_button = QPushButton("New")
        new_button.setMaximumWidth(60)
        new_button.clicked.connect(self.create_treatment)
        
        edit_button = QPushButton("Edit")
        edit_button.setMaximumWidth(60)
        edit_button.clicked.connect(self.edit_treatment)
        rename_button = QPushButton("Rename")
        rename_button.setMaximumWidth(60)
        rename_button.clicked.connect(self.rename_treatment)
        
        delete_button = QPushButton("Delete")
        delete_button.setMaximumWidth(60)
        delete_button.clicked.connect(self.delete_treatment)

        treatment_layout.addWidget(new_button)
        treatment_layout.addWidget(edit_button)
        treatment_layout.addWidget(rename_button)
        treatment_layout.addWidget(delete_button)

        main_layout.addLayout(treatment_layout)

        self.prompt_entry = QTextEdit()
        self.prompt_entry.setPlaceholderText("Enter Prompt Here...")
        self.prompt_entry.setMaximumHeight(80)
        self.prompt_entry.installEventFilter(self)
        main_layout.addWidget(self.prompt_entry)

        history_layout = QHBoxLayout()
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_label = QLabel("History:")
        self.history_combobox = QComboBox()
        self.history_combobox.addItems(self.prompt_history)
        self.history_combobox.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.history_combobox.currentIndexChanged.connect(self.load_selected_prompt)
        history_layout.addWidget(history_label)
        history_layout.addWidget(self.history_combobox)
        history_layout.addStretch()
        main_layout.addLayout(history_layout)

        enhance_layout = QHBoxLayout()
        self.grok_checkbox = QCheckBox("Enhance with GROK")
        self.text_model_combobox = QComboBox()
        enhance_button = QPushButton("Enhance Prompt")
        enhance_button.clicked.connect(self.enhance_prompt)
        revert_button = QPushButton("Revert Prompt")
        revert_button.clicked.connect(self.revert_to_original_prompt)
        enhance_layout.addWidget(self.grok_checkbox)
        enhance_layout.addWidget(self.text_model_combobox)
        enhance_layout.addWidget(enhance_button)
        enhance_layout.addWidget(revert_button)
        enhance_layout.addStretch()
        main_layout.addLayout(enhance_layout)

        config_layout = QGridLayout()
        config_layout.setColumnStretch(1, 1)
        
        seed_label = QLabel("Seed:")
        self.seed_entry = QLineEdit("-1")
        self.seed_entry.setFixedWidth(60)
        config_layout.addWidget(seed_label, 0, 0)
        config_layout.addWidget(self.seed_entry, 0, 1)

        ratio_label = QLabel("Aspect Ratio:")
        self.ratio_combobox = QComboBox()
        self.ratio_combobox.addItems(["1:1", "3:4", "16:9", "Custom"])
        self.ratio_combobox.setFixedWidth(80)
        self.ratio_combobox.currentIndexChanged.connect(self.toggle_custom_ratio)
        config_layout.addWidget(ratio_label, 0, 2)
        config_layout.addWidget(self.ratio_combobox, 0, 3)

        model_label = QLabel("Model:")
        self.model_combobox = QComboBox()
        self.model_combobox.addItems(["Default", "Turbo"])
        self.model_combobox.setFixedWidth(80)
        config_layout.addWidget(model_label, 0, 4)
        config_layout.addWidget(self.model_combobox, 0, 5)

        custom_width_label = QLabel("Width:")
        self.custom_width_entry = QLineEdit()
        self.custom_width_entry.setPlaceholderText("Width")
        self.custom_width_entry.setEnabled(False)
        config_layout.addWidget(custom_width_label, 1, 2)
        config_layout.addWidget(self.custom_width_entry, 1, 3)

        custom_height_label = QLabel("Height:")
        self.custom_height_entry = QLineEdit()
        self.custom_height_entry.setPlaceholderText("Height")
        self.custom_height_entry.setEnabled(False)
        config_layout.addWidget(custom_height_label, 1, 4)
        config_layout.addWidget(self.custom_height_entry, 1, 5)

        main_layout.addLayout(config_layout)

        button_layout = QHBoxLayout()

        generate_button = QPushButton("Generate")
        generate_button.clicked.connect(self.on_generate_button_click)
        button_layout.addWidget(generate_button)

        image_to_prompt_button = QPushButton("Image to Prompt")
        image_to_prompt_button.clicked.connect(self.on_image_to_prompt_click)
        button_layout.addWidget(image_to_prompt_button)

        paste_image_button = QPushButton("Paste Image")
        paste_image_button.clicked.connect(self.paste_image_from_clipboard)
        button_layout.addWidget(paste_image_button)

        select_image_button = QPushButton("🔍")
        select_image_button.setToolTip("Select an image from your computer")
        select_image_button.clicked.connect(self.select_and_display_image)
        select_image_button.setFixedSize(30, 30)
        button_layout.addWidget(select_image_button)
        
        viewer_button = QPushButton("VIEWER")
        viewer_button.setFixedWidth(80)
        viewer_button.clicked.connect(self.show_thumbnail_viewer)
        button_layout.addWidget(viewer_button)

        main_layout.addLayout(button_layout)

        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        self.image_label = QLabel()
        self.image_label.setFixedSize(520, 520)
        self.image_label.setStyleSheet("QLabel { background-color : white; }")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.mousePressEvent = self.image_label_mouse_press_event
        main_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        self.thumbnail_viewer = ThumbnailViewer(self)
        self.thumbnail_viewer.setParent(self)
        self.thumbnail_viewer.setWindowFlags(self.thumbnail_viewer.windowFlags() | Qt.Window)
        self.thumbnail_viewer.hide()  # Initially hide the thumbnail viewer

        self.status_bar = QLabel("Ready")
        self.status_bar.setAlignment(QtCore.Qt.AlignLeft)
        self.status_bar.setFixedHeight(20)
        main_layout.addWidget(self.status_bar)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        return main_layout

    def paste_image_from_clipboard(self):
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()

        if mime_data.hasImage():
            image = QImage(mime_data.imageData())
            if not image.isNull():
                pixmap = QPixmap.fromImage(image)
                scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
                self.status_bar.setText("Image pasted from clipboard")
                
                # Save the pasted image as the last generated image
                last_image_path = os.path.join(self.save_path, "last_generated_image.png")
                pixmap.save(last_image_path)

                # Convert QImage to PIL Image for consistency with other parts of the app
                buffer = QBuffer()
                buffer.open(QBuffer.ReadWrite)
                image.save(buffer, "PNG")
                pil_image = Image.open(io.BytesIO(buffer.data()))

                # Update the self.image attribute
                self.image = pil_image

                # Update last_loaded_image_path
                self.last_loaded_image_path = last_image_path

                self.status_bar.setText("Image pasted and ready for processing")
            else:
                self.status_bar.setText("Failed to load image from clipboard")
        else:
            self.status_bar.setText("No image found in clipboard")
        
    async def fetch_models_async(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://image.pollinations.ai/models') as response:
                    if response.status == 200:
                        self.models = await response.json()
                        self.models.insert(0, "No model")  # Add "No model" as the first option
                        self.update_model_combobox()
        except Exception as e:
            print(f"Error fetching models: {e}")

    def fetch_models(self):
        self.image_thread = QThread()
        self.image_worker = AsyncThread('https://image.pollinations.ai/models')
        self.image_worker.moveToThread(self.image_thread)
        self.image_worker.finished.connect(self.on_image_models_fetched)
        self.image_thread.started.connect(self.image_worker.run)
        self.image_thread.start()

        self.text_thread = QThread()
        self.text_worker = AsyncThread('https://text.pollinations.ai/models')
        self.text_worker.moveToThread(self.text_thread)
        self.text_worker.finished.connect(self.on_text_models_fetched)
        self.text_thread.started.connect(self.text_worker.run)
        self.text_thread.start()

    def on_image_models_fetched(self, models):
        if isinstance(models, dict):
            self.image_models = ['No model'] + list(models.keys())
        else:
            self.image_models = ['No model'] + models
        self.update_image_model_combobox()

    def on_text_models_fetched(self, models):
        print(f"Raw text models data: {models}")  # Debug print
        self.text_models = self.flatten_models(models)
        print(f"Flattened text models: {self.text_models}")  # Debug print
        self.update_text_model_combobox()

    def flatten_models(self, models):
        if isinstance(models, list) and all(isinstance(item, dict) for item in models):
            return [model['name'] for model in models if 'name' in model]
        elif isinstance(models, dict):
            return list(models.keys())
        elif isinstance(models, list):
            return models
        return []

    def update_text_model_combobox(self):
        self.text_model_combobox.clear()
        if self.text_models:
            self.text_model_combobox.addItems(self.text_models)
            self.text_model_combobox.setCurrentIndex(0)
        else:
            print("No text models available")
        print(f"Text model combobox updated with {self.text_model_combobox.count()} items")  # Debug print

    def create_treatment(self):
        dialog = TreatmentDialog(self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            pre_prompt, post_prompt = dialog.get_results()
            
            # Ask the user for a name
            name, ok = QtWidgets.QInputDialog.getText(self, "New Treatment", "Enter a name for the new treatment:")
            
            if ok and name:
                # Check if the name already exists
                while name in self.treatments:
                    QtWidgets.QMessageBox.warning(self, "Name Exists", "This name already exists. Please choose a different name.")
                    name, ok = QtWidgets.QInputDialog.getText(self, "New Treatment", "Enter a name for the new treatment:")
                    if not ok:
                        return  # User cancelled
                
                self.treatments[name] = {"pre": pre_prompt, "post": post_prompt}
                self.save_treatments(self.treatments)
                self.update_treatment_combobox()
                self.treatment_combobox.setCurrentText(name)  # Select the new treatment
            else:
                QtWidgets.QMessageBox.information(self, "Cancelled", "New treatment creation was cancelled.")

    def edit_treatment(self):
        name = self.treatment_combobox.currentText()
        if name != "None":
            treatment = self.treatments.get(name, {"pre": "", "post": ""})
            dialog = TreatmentDialog(self, initial_pre=treatment["pre"], initial_post=treatment["post"])
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                pre_prompt, post_prompt = dialog.get_results()
                self.treatments[name] = {"pre": pre_prompt, "post": post_prompt}
                self.save_treatments(self.treatments)

    def delete_treatment(self):
        name = self.treatment_combobox.currentText()
        if name != "None":
            reply = QtWidgets.QMessageBox.question(self, 'Delete Treatment', f"Are you sure you want to delete the treatment '{name}'?",
                                                   QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.Yes:
                del self.treatments[name]
                self.save_treatments(self.treatments)
                self.update_treatment_combobox()

    def load_selected_prompt(self):
        selected_prompt = self.history_combobox.currentText()
        if selected_prompt:
            self.prompt_entry.setText(selected_prompt)
        else:
            self.prompt_entry.clear()

    def toggle_custom_ratio(self):
        if self.ratio_combobox.currentText() == "Custom":
            self.custom_width_entry.setEnabled(True)
            self.custom_height_entry.setEnabled(True)
        else:
            self.custom_width_entry.setEnabled(False)
            self.custom_height_entry.setEnabled(False)

    def toggle_on_top(self):
        self.on_top = not self.on_top
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, self.on_top)
        self.show()  # This is necessary to apply the window flag change
        self.on_top_action.setChecked(self.on_top)
        self.save_settings()
        print(f"On Top toggled: {self.on_top}")  # Debug print

    def set_treatments_path(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Treatments Directory")
        if path:
            self.treatments_path = path
            self.save_settings()
            self.treatments = self.sync_treatments()
            self.update_treatment_combobox()
            self.status_bar.setText(f"Treatments path set to: {path}")

    async def async_generate_image(self, url):
        print(f"Attempting to generate image from URL: {url}")
        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        if response.status == 200:
                            content = await response.read()
                            print(f"Image data received, processing...")
                            image = Image.open(io.BytesIO(content))
                            print(f"Image loaded successfully")
                            return image
                        else:
                            self.status_bar.setText(f"Error {response.status}. Retrying... (Attempt {attempt + 1})")
                            print(f"Error {response.status} on attempt {attempt + 1}")
            except Exception as e:
                self.status_bar.setText(f"Network error. Retrying... (Attempt {attempt + 1})")
                print(f"Network error on attempt {attempt + 1}: {e}")

            if attempt < 2:
                await asyncio.sleep(5)

        self.status_bar.setText("Failed to generate image after multiple attempts. Please try again.")
        print("Failed to generate image after multiple attempts.")
        return None

    def enhance_prompt(self):
        current_prompt = self.prompt_entry.toPlainText().strip()
        if current_prompt:
            self.original_prompt = current_prompt
            selected_model = self.text_model_combobox.currentText()
            self.status_bar.setText(f"Enhancing prompt with {selected_model}...")
            threading.Thread(target=self.enhance_thread, args=(current_prompt, selected_model)).start()
        else:
            self.status_bar.setText("Error: Please enter a valid prompt.")

    def enhance_thread(self, prompt, model):
        enhanced_prompt = self.prompt_enhancer.enhance_prompt(prompt, model)
        self.update_prompt_signal.emit(enhanced_prompt)

    @pyqtSlot(str)
    def update_prompt_entry(self, text):
        self.prompt_entry.setText(text)
        self.status_bar.setText("Prompt enhanced")
        print(f"Prompt updated to: {text}")  # Debugging line

    def revert_to_original_prompt(self):
        if hasattr(self, 'original_prompt'):
            self.prompt_entry.setText(self.original_prompt)
            self.status_bar.setText("Reverted to original prompt")
        else:
            self.status_bar.setText("No original prompt to revert to")
        
    def generate_image(self):
        if self.generating_image:
            return
        self.generating_image = True

        try:
            current_prompt = self.prompt_entry.toPlainText().strip()
            if not current_prompt:
                self.status_bar.setText("Error: Please enter a valid prompt.")
                return

            selected_treatment = self.treatment_combobox.currentText()
            treatment = self.treatments.get(selected_treatment, {"pre": "", "post": ""})
            full_prompt = f"{treatment['pre']} {current_prompt} {treatment['post']}".strip()

            seed = self.seed_entry.text().strip()
            if not seed.isdigit() and seed != "-1":
                self.status_bar.setText("Error: Please enter a valid seed or -1 for random.")
                return

            if seed == "-1":
                seed = str(random.randint(0, 99999))
            self.last_seed = seed

            model = self.model_combobox.currentText()
            if model == "No model":
                model = ""  # Use empty string for no model
                
            aspect_ratio = self.ratio_combobox.currentText()
            width, height = {"1:1": ("1024", "1024"), "3:4": ("768", "1024"), "16:9": ("1024", "576")}.get(aspect_ratio, ("1024", "1024"))
            if aspect_ratio == "Custom":
                width = self.custom_width_entry.text().strip()
                height = self.custom_height_entry.text().strip()
                if not width.isdigit() or not height.isdigit():
                    self.status_bar.setText("Error: Please enter valid dimensions.")
                    return

            # Prepare the URL
            url = f"https://pollinations.ai/p/{urllib.parse.quote(full_prompt)}?seed={seed}&width={width}&height={height}&nologo=True&nofeed=true"

            if model:
                url += f"&model={model}"
            
            if self.grok_checkbox.isChecked():
                url += "&enhance=true"

            # Start a thread to generate the image without blocking the UI
            threading.Thread(target=self.download_and_display_image, args=(url,)).start()

        except Exception as e:
            self.status_bar.setText(f"Error in generate_image: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.generating_image = False

    def pil_to_qimage(self, pil_image):
        try:
            if isinstance(pil_image, QPixmap):
                return pil_image.toImage()
            img_array = np.array(pil_image.convert("RGB"))
            height, width, channel = img_array.shape
            bytes_per_line = 3 * width
            return QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        except Exception as e:
            print(f"Error in pil_to_qimage: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def on_generate_button_click(self):
        prompt = self.prompt_entry.toPlainText().strip()
        print(f"Current prompt: '{prompt}'")
        print(f"Current treatment: '{self.treatment_combobox.currentText()}'")
        if prompt:
            if prompt not in self.prompt_history:
                self.prompt_history.insert(0, prompt)
                self.prompt_history = self.prompt_history[:10]
                self.save_prompt_history()
                self.update_history_combobox()

            # Store the current seed
            current_seed = self.seed_entry.text().strip()

            # Call generate_image using threading to avoid blocking the GUI
            threading.Thread(target=self.generate_image).start()

            # Restore the seed after generation starts
            self.seed_entry.setText(current_seed)
        else:
            self.status_bar.setText("Error: Please enter a valid prompt.")

    def copy_image_to_clipboard(self, image):
        if isinstance(image, QImage):
            clipboard = QApplication.clipboard()
            clipboard.setImage(image)
            self.status_bar.setText("Image copied to clipboard")
        else:
            self.status_bar.setText("Failed to copy image to clipboard")

    def save_image(self, image):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        save_path = os.path.join(self.save_path, f"Image-{timestamp}.png")
        
        last_image_path = os.path.join(self.save_path, "last_generated_image.png")
        
        if isinstance(image, QImage):
            image.save(save_path)
            image.save(last_image_path)
        else:
            image.save(save_path)
            image.save(last_image_path)
        
        self.status_bar.setText(f"Image saved to: {save_path}")
        print(f"Image saved to: {save_path}")
        
        # Use invokeMethod to update thumbnail in the main thread
        QMetaObject.invokeMethod(self, "update_thumbnail", Qt.QueuedConnection, Q_ARG(str, save_path))
        
        return save_path

    def load_settings(self):
        try:
            with open('settings.json', 'r') as f:
                settings = json.load(f)
                self.on_top = settings.get('on_top', False)
                self.treatments_path = settings.get('treatments_path', '')
                self.on_top_action.setChecked(self.on_top)
                self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, self.on_top)
                self.show()  # This is necessary to apply the window flag change
        except FileNotFoundError:
            pass

    def save_settings(self):
        new_settings = {
            'on_top': self.on_top,
            'treatments_path': self.treatments_path
        }
        try:
            with open('settings.json', 'r') as f:
                old_settings = json.load(f)
        except FileNotFoundError:
            old_settings = {}
        
        if new_settings != old_settings:
            with open('settings.json', 'w') as f:
                json.dump(new_settings, f)
            print(f"Settings saved: {new_settings}")  # Debug print
        else:
            print("Settings unchanged, not saving")

    def sync_treatments(self):
        local_treatments = self.load_local_treatments()
        if self.treatments_path:
            network_treatments = self.load_network_treatments()
            merged_treatments = {**local_treatments, **network_treatments}
            self.save_treatments(merged_treatments)
            return merged_treatments
        return local_treatments

    def load_local_treatments(self):
        try:
            with open('treatments.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def load_network_treatments(self):
        try:
            network_file = os.path.join(self.treatments_path, 'treatments.json')
            if os.path.exists(network_file):
                with open(network_file, 'r') as f:
                    return json.load(f)
            else:
                print(f"Network treatments file not found: {network_file}")
                return {}
        except Exception as e:
            print(f"Error loading network treatments: {e}")
            return {}

    def save_treatments(self, treatments):
        with open('treatments.json', 'w') as f:
            json.dump(treatments, f, indent=2)

        if self.treatments_path:
            try:
                os.makedirs(self.treatments_path, exist_ok=True)
                network_file = os.path.normpath(os.path.join(self.treatments_path, 'treatments.json'))
                with open(network_file, 'w') as f:
                    json.dump(treatments, f, indent=2)
                print(f"Treatments saved to network location: {network_file}")
            except Exception as e:
                print(f"Error saving treatments to network location: {e}")

    def load_prompt_history(self):
        try:
            with open(self.prompt_history_file, 'r') as f:
                data = json.load(f)
                return [item['prompt'] for item in data.values() if 'prompt' in item]
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def save_prompt_history(self):
        data = {str(i): {'prompt': prompt} for i, prompt in enumerate(self.prompt_history)}
        with open(self.prompt_history_file, 'w') as f:
            json.dump(data, f)

    def update_treatment_combobox(self):
        current_text = self.treatment_combobox.currentText()
        self.treatment_combobox.clear()
        self.treatment_combobox.addItems(["None"] + list(self.treatments.keys()))
        index = self.treatment_combobox.findText(current_text)
        if index >= 0:
            self.treatment_combobox.setCurrentIndex(index)
        else:
            self.treatment_combobox.setCurrentIndex(0)
            
    def update_history_combobox(self):
        self.history_combobox.clear()
        self.history_combobox.addItems(self.prompt_history)

    def on_closing(self):
        self.save_settings()
        self.close()

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)
    
    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Return and event.modifiers() & QtCore.Qt.ShiftModifier:
                cursor = self.prompt_entry.textCursor()
                cursor.insertText('\n')
                return True
            elif event.key() == QtCore.Qt.Key_Return:
                self.on_generate_button_click()
                return True
        return super().eventFilter(source, event)

    def image_label_mouse_press_event(self, event):
        if event.button() == Qt.LeftButton:
            self.show_larger_image()
        elif event.button() == Qt.RightButton:
            self.show_context_menu(event.pos())

    def show_context_menu(self, position):
        if self.image_label.pixmap():
            context_menu = QMenu(self)
            copy_action = context_menu.addAction("Copy Image")
            action = context_menu.exec_(self.image_label.mapToGlobal(position))
            if action == copy_action:
                self.copy_image_to_clipboard(self.image_label.pixmap().toImage())

    def show_larger_image(self):
        if self.image_label.pixmap():
            if hasattr(self, 'enlarged_image_dialog') and self.enlarged_image_dialog.isVisible():
                self.enlarged_image_dialog.close()
            else:
                self.enlarged_image_dialog = QDialog(self)
                self.enlarged_image_dialog.setWindowTitle("Enlarged Image")
                self.enlarged_image_dialog.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
                layout = QVBoxLayout()
                label = ClickableLabel()
                
                # Calculate the size for the enlarged image (80% of screen height)
                screen = QApplication.primaryScreen().geometry()
                target_height = int(screen.height() * 0.8)
                scaled_pixmap = self.image_label.pixmap().scaled(
                    target_height, target_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                label.setPixmap(scaled_pixmap)
                layout.addWidget(label)
                self.enlarged_image_dialog.setLayout(layout)

                # Connect the click event to close the dialog
                label.clicked.connect(self.enlarged_image_dialog.close)

                self.enlarged_image_dialog.show()

    def rename_treatment(self):
        current_name = self.treatment_combobox.currentText()
        if current_name != "None":
            new_name, ok = QInputDialog.getText(self, "Rename Treatment", "Enter new name:", text=current_name)
            if ok and new_name:
                if new_name in self.treatments:
                    QMessageBox.warning(self, "Error", "A treatment with this name already exists.")
                else:
                    self.treatments[new_name] = self.treatments.pop(current_name)
                    self.save_treatments(self.treatments)
                    self.update_treatment_combobox()
                    self.treatment_combobox.setCurrentText(new_name)

class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

if __name__ == "__main__":
    try:
        app = QApplication([])
        window = ImageGeneratorApp()
        window.show()

        def clean_exit():
            window.cleanup()
            app.quit()

        app.aboutToQuit.connect(clean_exit)
        sys.exit(app.exec_())
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

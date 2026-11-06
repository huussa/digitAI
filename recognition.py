import sys
from pathlib import Path

import numpy as np
import pygame
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "model.keras"

# Model can be supplied explicitly, otherwise use digits/model.keras.
if len(sys.argv) > 2:
    sys.exit("Usage: python recognition.py [model.keras]")

if len(sys.argv) == 2:
    model_path = Path(sys.argv[1]).expanduser()
    if not model_path.is_absolute():
        # First try the path exactly as typed from the current working directory.
        candidate = (Path.cwd() / model_path).resolve()
        # If that does not exist, also try next to recognition.py.
        model_path = candidate if candidate.exists() else (BASE_DIR / model_path).resolve()
else:
    model_path = DEFAULT_MODEL

if not model_path.exists():
    sys.exit(
        f"Model not found: {model_path}\n\n"
        "First train the model:\n"
        "  python handwriting.py\n\n"
        "Then run:\n"
        "  python recognition.py"
    )

model = tf.keras.models.load_model(model_path, compile=False)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Start pygame
pygame.init()
size = width, height = 600, 400
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Digit Recognition")

# The 2023 source expected an assets/font file that is not included in this archive.
# Use a system font instead so the project works without external assets.
smallFont = pygame.font.SysFont(None, 20)
largeFont = pygame.font.SysFont(None, 40)

ROWS, COLS = 28, 28
OFFSET = 20
CELL_SIZE = 10

handwriting = [[0] * COLS for _ in range(ROWS)]
classification = None

while True:
    # Check if game quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill(BLACK)

    # Check for mouse press
    click, _, _ = pygame.mouse.get_pressed()
    mouse = pygame.mouse.get_pos() if click == 1 else None

    # Draw each grid cell
    for i in range(ROWS):
        for j in range(COLS):
            rect = pygame.Rect(
                OFFSET + j * CELL_SIZE,
                OFFSET + i * CELL_SIZE,
                CELL_SIZE, CELL_SIZE
            )

            # If cell has been written on, darken cell
            if handwriting[i][j]:
                channel = int(255 - (handwriting[i][j] * 255))
                pygame.draw.rect(screen, (channel, channel, channel), rect)
            else:
                pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

            # If writing on this cell, fill in current cell and neighbors
            if mouse and rect.collidepoint(mouse):
                handwriting[i][j] = 250 / 255
                if i + 1 < ROWS:
                    handwriting[i + 1][j] = 220 / 255
                if j + 1 < COLS:
                    handwriting[i][j + 1] = 220 / 255
                if i + 1 < ROWS and j + 1 < COLS:
                    handwriting[i + 1][j + 1] = 190 / 255

    # Reset button
    resetButton = pygame.Rect(30, OFFSET + ROWS * CELL_SIZE + 30, 100, 30)
    resetText = smallFont.render("Reset", True, BLACK)
    resetTextRect = resetText.get_rect(center=resetButton.center)
    pygame.draw.rect(screen, WHITE, resetButton)
    screen.blit(resetText, resetTextRect)

    # Classify button
    classifyButton = pygame.Rect(150, OFFSET + ROWS * CELL_SIZE + 30, 100, 30)
    classifyText = smallFont.render("Classify", True, BLACK)
    classifyTextRect = classifyText.get_rect(center=classifyButton.center)
    pygame.draw.rect(screen, WHITE, classifyButton)
    screen.blit(classifyText, classifyTextRect)

    # Reset drawing
    if mouse and resetButton.collidepoint(mouse):
        handwriting = [[0] * COLS for _ in range(ROWS)]
        classification = None

    # Generate classification
    if mouse and classifyButton.collidepoint(mouse):
        image = np.array(handwriting, dtype=np.float32).reshape(1, 28, 28, 1)
        probabilities = model.predict(image, verbose=0)
        classification = int(np.argmax(probabilities[0]))

    # Show classification if one exists
    if classification is not None:
        classificationText = largeFont.render(str(classification), True, WHITE)
        classificationRect = classificationText.get_rect()
        grid_size = OFFSET * 2 + CELL_SIZE * COLS
        classificationRect.center = (grid_size + ((width - grid_size) / 2), 100)
        screen.blit(classificationText, classificationRect)

    pygame.display.flip()

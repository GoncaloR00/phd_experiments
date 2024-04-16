#!/usr/bin/env python3

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import random

def generate_pdf(filename, num_points):
    width, height = A4

    # Create a canvas object
    c = canvas.Canvas(filename, pagesize=A4)

    # Draw points with random sizes and locations
    for _ in range(num_points):
        # Generate random coordinates within the page boundaries
        x = random.uniform(50, width - 50)
        y = random.uniform(50, height - 50)

        # Generate random point size
        size = random.uniform(1, 3)

        # Draw the point
        c.circle(x, y, size, stroke=1, fill=1)

    # Save the canvas to a PDF file
    c.save()

# Usage example
generate_pdf("random_points.pdf", 5000)
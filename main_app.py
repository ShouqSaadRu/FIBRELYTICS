import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
from backend import (
    process_image_pipeline, save_fiber_data
)
from tkinter import ttk
import sv_ttk  


COLORS = {
    'primary': '#4a6fa5',
    'secondary': '#166088',
    'background': '#f0f4f8',
    'text': '#333333',
    'accent': '#4fc3f7',
    'success': '#4caf50',
    'warning': '#ff9800',
    'error': '#f44336'
}

FONTS = {
    'title': ('Segoe UI', 14, 'bold'),
    'subtitle': ('Segoe UI', 11),
    'body': ('Segoe UI', 10),
    'button': ('Segoe UI', 8, 'bold')
}

def open_image():
    global original_img
    global polygon_mask
    polygon_mask = None
    path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
    if path:
        original_img = cv2.imread(path)
        show_preview("Original", original_img)
        status_label.config(text="Image loaded. Ready to process.")
        polygon_btn.pack(pady=5)

def show_preview(step_name, img, contours=None, centers=None, all_rays=None):
    
    try:
        # Create a working copy
        display_img = img.copy()
        
        # Handle different image types
        if len(display_img.shape) == 2 or display_img.shape[2] == 1:
            # Grayscale or binary image
            if img.dtype == np.bool_ or np.max(display_img) == 1:
                # Binary threshold image
                display_img = (display_img * 255).astype(np.uint8)
            else:
                # Normal grayscale - normalize with 1% percentile clipping
                low = np.percentile(display_img, 1)
                high = np.percentile(display_img, 99)
                display_img = np.clip(display_img, low, high)
                display_img = ((display_img - low) / (high - low + 1e-5) * 255).astype(np.uint8)
            
            # Convert to RGB for visualization
            display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
        
        elif display_img.dtype != np.uint8:
            # Color image with wrong dtype
            display_img = np.clip(display_img, 0, 255).astype(np.uint8)
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        else:
            # Proper color image
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
        
        
        # Draw rays if provided
        if all_rays is not None:
            for center, rays in all_rays:
                yc, xc = int(center[0]), int(center[1])
                for (x, y) in rays:
                    cv2.line(display_img, (xc, yc), (int(x), int(y)), (255, 0, 128), 1)
        
        # Draw contours and centers if provided
        if contours is not None and centers is not None:
            display_img = display_img.copy()
            cv2.drawContours(display_img, contours, -1, (107, 255, 83), 2)
            for idx, (y, x) in enumerate(centers):
                cx, cy = int(x), int(y)
                cv2.circle(display_img, (cx, cy), 2, (255, 0, 0), -1)
                if len(centers) < 45:
                    cv2.putText(display_img, str(idx + 1), (cx + 4, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    cv2.putText(display_img, str(idx + 1), (cx + 4, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        # Convert to PIL format and resize
        img_pil = Image.fromarray(display_img).resize((350, 350))
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # Update the display
        image_label.config(image=img_tk)
        image_label.image = img_tk
        status_label.config(text=f"Now showing: {step_name}")
        
    except Exception as e:
        print(f"Error in show_preview: {str(e)}")
        status_label.config(text=f"Error displaying {step_name}")

def show_preview_mask(step_name, img):
    try:
        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(display_img).resize((350, 350))
        img_tk = ImageTk.PhotoImage(img_pil)
        mask_label.config(image=img_tk)
        mask_label.image = img_tk
        status_label.config(text=f"Now showing: {step_name}")
    except Exception as e:
        print(f"Error in show_preview_mask: {str(e)}")
        status_label.config(text=f"Error showing mask preview")

def process_steps():
    global processed_result
    if original_img is None:
        messagebox.showerror("Error", "Please upload an image first.")
        return

    try:
        progress.pack(fill="x", pady=(5, 0))
        progress['value'] = 0
        status_label.config(text="Processing started...")
        app.update()
        
        params = {
            "sp": int(sp_entry.get()),
            "sr": int(sr_entry.get()),
            "clip_limit": float(clip_entry.get()),
            "tile_grid_size": (int(tile_x_entry.get()), int(tile_y_entry.get())),
            "blur_size": int(blur_entry.get()),
            "threshold_override": None,
            "min_distance": int(min_dist_entry.get()),
            "threshold": float(proximity_thresh_entry.get())
        }
        
        result = process_image_pipeline(original_img, params, mask=polygon_mask)

        preview_steps = ["Original","Mean Shift", "Blurred","Gray scale", "Threshold", "Watershed"]
        steps = len(preview_steps)
        
        for i, key in enumerate(preview_steps):
            show_preview(key, result[key])
            progress['value'] = (i+1)/steps * 100
            status_label.config(text=f"Processing: {key}...")
            app.update()
            time.sleep(1)  
        
        show_preview("Radius Detected", original_img, 
                    result["Contours"], 
                    result["Fibre Centers"],
                    result["all_rays"])
        
        fv_label.config(text=f"Fiber Volume Fraction: {result['V_fv']:.4f}%")
        processed_result = result
        download_btn.config(state="normal")
        status_label.config(text="Processing complete!")
        
    except Exception as e:
        messagebox.showerror("Processing Error", str(e))
        status_label.config(text="Error in processing")
    finally:
        progress.pack_forget()




def download_dataset():
    if processed_result is None:
        messagebox.showwarning("Warning", "No dataset to save. Please process an image first.")
        return
    
    path = filedialog.asksaveasfilename(defaultextension=".csv",
                                         filetypes=[("CSV Files", "*.csv")],
                                         title="Save Dataset As")
    if path:
        try:
            save_fiber_data(processed_result, path)
            messagebox.showinfo("Success", "Dataset saved successfully.")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))


def draw_polygon(image):
    global polygon_mask, original_img

    if image is None:
        messagebox.showerror("Error", "Upload an image first.")
        return

    drawing = False
    points = []
    temp_img = image.copy()

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(temp_img, (x, y), 3, (0, 255, 255), -1)
            if len(points) > 1:
                cv2.line(temp_img, points[-2], points[-1], (0, 255, 255), 2)
            cv2.imshow("Draw Tape Boundary - Press Q when done", temp_img)

    cv2.imshow("Draw Tape Boundary - Press Q when done", temp_img)
    cv2.setMouseCallback("Draw Tape Boundary - Press Q when done", click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    if len(points) < 3:
        messagebox.showwarning("Warning", "Polygon requires at least 3 points.")
        return

    # Create the binary mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    polygon_mask = mask
    
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    show_preview_mask("Masked Polygon Applied", masked_img)
    process_steps()
    
    
# UI 
app = tk.Tk()
app.title("Fibrelytics - Composite Analysis")
app.geometry("1200x700")
app.configure(bg=COLORS['background'])

# Apply modern theme
sv_ttk.set_theme("light")  # or "dark" for dark mode

# Main container with modern styling
main_container = tk.Frame(app, bg=COLORS['background'], padx=15, pady=15)
main_container.pack(fill="both", expand=True)

# Left panel with card-like appearance
control_panel = tk.Frame(main_container, bg='white', bd=0, highlightthickness=0, 
                        relief="solid", padx=15, pady=15)
control_panel.pack(side="left", fill="y", padx=(0, 15))

# Right panel with card-like appearance
display_panel = tk.Frame(main_container, bg='white', bd=0, highlightthickness=0,
                        relief="solid", padx=15, pady=15)
display_panel.pack(side="right", fill="both", expand=True)

# Control Panel Contents ==========================================
tk.Label(control_panel, text="FIBRE ANALYSIS", font=FONTS['title'], 
        bg='white', fg=COLORS['primary']).pack(pady=(0, 15))

# Section header
params_header = tk.Frame(control_panel, bg='white')
params_header.pack(fill="x", pady=(0, 10))
tk.Label(params_header, text="PROCESSING PARAMETERS", font=FONTS['subtitle'],
        bg='white', fg=COLORS['secondary']).pack(side="left")

# Modern parameter entries with tooltips
def add_param(parent, label, default="", tooltip=None):
    frame = tk.Frame(parent, bg='white', pady=4)
    frame.pack(fill="x")
    
    tk.Label(frame, text=label, font=FONTS['body'], bg='white', 
            fg=COLORS['text'], anchor="w").pack(side="left")
    
    entry = ttk.Entry(frame, width=12, font=FONTS['body'])
    entry.insert(0, default)
    entry.pack(side="right")
    
    if tooltip:
        tooltip_label = tk.Label(frame, text="?", font=FONTS['body'], 
                               fg=COLORS['accent'], bg='white', cursor="question_arrow")
        tooltip_label.pack(side="right", padx=5)
        tooltip_label.bind("<Enter>", lambda e: show_tooltip(tooltip))
        tooltip_label.bind("<Leave>", lambda e: hide_tooltip())
    
    return entry

# Add tooltip functionality
def show_tooltip(text):
    tooltip = tk.Toplevel(app)
    tooltip.wm_overrideredirect(True)
    tooltip.wm_geometry(f"+{app.winfo_pointerx()+10}+{app.winfo_pointery()+10}")
    label = tk.Label(tooltip, text=text, bg="#ffffe0", relief="solid", 
                    borderwidth=1, font=FONTS['body'], padx=5, pady=5)
    label.pack()
    tooltip.tooltip_label = label
    app.tooltip = tooltip

def hide_tooltip():
    if hasattr(app, 'tooltip'):
        app.tooltip.destroy()

# Add parameters with tooltips
sp_entry = add_param(control_panel, "Mean Shift - sp", "30", "Spatial window radius")
sr_entry = add_param(control_panel, "Mean Shift - sr", "50", "Color radius")
clip_entry = add_param(control_panel, "CLAHE Clip Limit", "2.0", "Contrast limit")
tile_x_entry = add_param(control_panel, "CLAHE Tile X", "8", "Grid size X dimension")
tile_y_entry = add_param(control_panel, "CLAHE Tile Y", "8", "Grid size Y dimension")
blur_entry = add_param(control_panel, "Gaussian Blur Size", "21", "Kernel size (odd number)")
min_dist_entry = add_param(control_panel, "Min Distance", "7", "Minimum fiber separation")
proximity_thresh_entry = add_param(control_panel, "Proximity Threshold", "5", "Distance threshold")

#buttons
button_frame = tk.Frame(control_panel, bg='white', pady=15)
button_frame.pack(fill="x")

def create_modern_button(parent, text, command, color=COLORS['primary']):
    btn = tk.Button(parent, text=text, command=command, 
                   font=FONTS['button'], bg=color, fg='white',
                   activebackground=color, activeforeground='white',
                   bd=0, padx=15, pady=2, relief="flat",
                   highlightthickness=0)
    btn.pack(fill="x", pady=4)
    return btn

upload_btn = create_modern_button(button_frame, "Upload Image", open_image)
process_btn = create_modern_button(button_frame, "Process Image", process_steps, COLORS['secondary'])
polygon_btn = create_modern_button(button_frame, "Draw Polygon", lambda: draw_polygon(original_img))
polygon_btn.pack_forget()
download_btn = create_modern_button(button_frame, "Download Dataset", download_dataset, COLORS['success'])
download_btn.config(state="disabled")


status_frame = tk.Frame(control_panel, bg='white')
status_frame.pack(fill="x", pady=(10, 0))

status_label = tk.Label(status_frame, text="Ready", font=FONTS['body'], 
                       bg='white', fg=COLORS['text'], anchor="w")
status_label.pack(fill="x")

# Progress bar (for processing steps)
progress = ttk.Progressbar(status_frame, orient="horizontal", length=100, mode="determinate")
progress.pack(fill="x", pady=(5, 0))
progress.pack_forget()  # Only show during processing

fv_label = tk.Label(status_frame, text="Fiber Volume: --", font=FONTS['body'], 
                   bg='white', fg=COLORS['primary'], anchor="w")
fv_label.pack(fill="x", pady=(5, 0))

# Display Panel Contents ==========================================
# Image display frames with cards
image_container = tk.Frame(display_panel, bg='white')
image_container.pack(fill="both", expand=True, pady=10)

def create_image_card(parent, title):
    card = tk.Frame(parent, bg='#f8f9fa', bd=0, highlightthickness=1,
                   highlightbackground='#e0e0e0', padx=10, pady=10)
    
    header = tk.Frame(card, bg='#f8f9fa')
    header.pack(fill="x", pady=(0, 10))
    tk.Label(header, text=title, font=FONTS['subtitle'], 
            bg='#f8f9fa', fg=COLORS['secondary']).pack(side="left")
    
    img_label = tk.Label(card, bg='white', bd=1, relief="solid")
    img_label.pack(fill="both", expand=True)
    
    return card, img_label

# Original image card
original_card, image_label = create_image_card(image_container, "ORIGINAL/PROCESSED")
original_card.pack(side="left", fill="both", expand=True, padx=5)

# Masked image card
masked_card, mask_label = create_image_card(image_container, "MASKED VIEW")
masked_card.pack(side="left", fill="both", expand=True, padx=5)

original_img = None
app.mainloop()
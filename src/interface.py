#!/usr/bin/env python
# -*- coding: utf-8 -*-

from imgui.integrations.glfw import GlfwRenderer
from testwindow import show_test_window
import OpenGL.GL as gl
import glfw
import imgui
import sys
import imageio
from PIL import Image
from skimage.transform import resize

import matplotlib.pyplot as plt

from gan import Generator
import load_model_gan
import load_model_vae

IMAGE_SIZE = 64


def update_texture(texture_id, image):
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)  # bind the texture
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D,
        0,
        gl.GL_RGB,
        image.shape[1],
        image.shape[0],
        0,
        gl.GL_RGB if image.shape[2] == 3 else gl.GL_RGBA,
        gl.GL_FLOAT,
        image,
    )  # re-upload the texture data
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)  # unbind the texture


def load_texture(image):
    width, height, channels = image.shape

    # Generate a new texture ID
    texture_id = gl.glGenTextures(1)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    # Upload the image data to GPU memory
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D,
        0,
        gl.GL_RGB,
        width,
        height,
        0,
        gl.GL_RGB if image.shape[2] == 3 else gl.GL_RGBA,
        gl.GL_FLOAT,
        image,
    )
    return texture_id


def main():
    imgui.create_context()
    window = impl_glfw_init()
    global impl
    impl = GlfwRenderer(window)

    fixed_names = []
    fixed_images = []
    fixed_textures = []
    # load fixed images
    for i in range(20):
        name = f"face_{i+1}"
        img = imageio.imread(f"../faces_output_pick/{i}.png")
        img = (img - img.min()) / (img.max() - img.min())
        fixed_names.append(name)
        fixed_images.append(resize(img, (IMAGE_SIZE, IMAGE_SIZE))[..., :3])
        fixed_textures.append(load_texture(img[..., :3]))

    image1_idx = 0
    image2_idx = 19

    model_idx = 0
    model_names = ["VAE", "GAN"]

    slider_idx = 0

    result_images = None
    result_textures = None

    vae = load_model_vae.load_vae()
    netG, noises = load_model_gan.load_model()

    result_images = load_model_vae.interp_vae(
        vae, fixed_images, 10, image1_idx, image2_idx
    )

    result_textures = [
        load_texture(result_images[i]) for i in range(len(result_images))
    ]

    show_custom_window = False
    show_image_select_window = True

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()

        # Make the next window take up the full frame
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(*imgui.get_io().display_size)

        if show_image_select_window:
            is_expand, show_image_select_window = imgui.begin(
                "Image select window",
                True,
                flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_TITLE_BAR,
            )
            if is_expand:

                def update_interpolate_image():
                    print("Model index:", model_idx)
                    # update interpolated images
                    if model_idx == 0:
                        result_images = load_model_vae.interp_vae(
                            vae, fixed_images, 10, image1_idx, image2_idx
                        )
                    else:
                        result_images = load_model_gan.interp_gan(
                            netG, noises, 10, image1_idx, image2_idx
                        )
                    nonlocal result_textures
                    result_textures = [
                        load_texture(result_images[i])
                        for i in range(len(result_images))
                    ]

                clicked, model_idx = imgui.combo("Model to use", model_idx, model_names)
                if clicked:
                    update_interpolate_image()

                imgui.text("Select two images to interpolate between:")
                # Create a dropdown menu and populate with image names
                clicked1, image1_idx = imgui.combo("Face 1", image1_idx, fixed_names)
                clicked2, image2_idx = imgui.combo("Face 2", image2_idx, fixed_names)
                if clicked1 or clicked2:
                    update_interpolate_image()

                # Render the selected image
                imgui.image(fixed_textures[image1_idx], 64, 64)
                imgui.same_line()
                imgui.image(fixed_textures[image2_idx], 64, 64)

                imgui.text("Result:")
                _, slider_idx = imgui.slider_int("integer slider", slider_idx, 0, 9)
                imgui.image(result_textures[slider_idx], 64, 64)

            imgui.end()

        if show_custom_window:
            is_expand, show_custom_window = imgui.begin("Custom window", True)
            if is_expand:
                imgui.text("Bar")
                imgui.text_ansi("B\033[31marA\033[mnsi ")
                imgui.text_ansi_colored("Eg\033[31mgAn\033[msi ", 0.2, 1.0, 0.0)
                imgui.extra.text_ansi_colored("Eggs", 0.2, 1.0, 0.0)
            imgui.end()

        # show_test_window()

        gl.glClearColor(0.0, 0.0, 0.0, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()


def impl_glfw_init():
    width, height = 1280, 720
    window_name = "minimal ImGui/GLFW3 example"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        sys.exit(1)

    # OS X supports only forward-compatible core profiles from 3.2
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(int(width), int(height), window_name, None, None)
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        sys.exit(1)

    return window


if __name__ == "__main__":
    main()

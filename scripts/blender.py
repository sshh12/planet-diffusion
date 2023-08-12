import bpy
import glob

material_name = "PlanetMaterial"
object_name = "Planet"

image_list = glob.glob(r"...")

mat = bpy.data.materials[material_name]
obj = bpy.data.objects[object_name]

for i, image_path in enumerate(image_list):
    if "-render.png" in image_path:
        continue
    output_path = image_path.replace(".png", "-render.png")
    img = bpy.data.images.load(image_path)

    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    tex_node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex_node.image = img
    mat.node_tree.links.new(bsdf.inputs["Base Color"], tex_node.outputs["Color"])

    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

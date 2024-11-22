from BVH_FILE import read_bvh


animation = read_bvh("/home/giuliamartinelli/Documents/SportTech/MoCap File/Prova1_Lorenzo.bvh")

anim_data, names = animation[0], animation[1]

offsets = anim_data.offsets
orients = anim_data.orients
parents = anim_data.parents
positions = anim_data.positions
rotations = anim_data.rotations





print('here')
import sys
import array 
from math import sqrt
import numpy as np

# SPHERE <name> <pos x> <pos y> <pos z> <scl x> <scl y> <scl z> <r> <g> <b> <Ka> <Kd> <Ks> <Kr> <n>
class Sphere:
    """Sphere is the only 3D shape implemented. Has center, radius and material"""

    def __init__(self, center, radius, material,scale,shinyfactor,lightproperties):
        self.translation = center
        self.radius = radius
        self.material = material
        self.scale = scale
        self.shiny = shinyfactor
        self.ambience = lightproperties[0]
        self.diffuse = lightproperties[1]
        self.specular = lightproperties[2]
        self.reflect = lightproperties[3]
        self.matrix = None
    #create the inverse matrix used for the transformed ray
    def create_scale_matrix(self): 
        matrix_to_invert = np.array([[self.scale[0],0,0,self.translation[0]],
                                    [0,self.scale[1],0,self.translation[1]],
                                    [0,0,self.scale[2],self.translation[2]],
                                    [0,0,0,1]])
        matrix_inverted = np.linalg.inv(matrix_to_invert) 
        self.matrix = matrix_inverted

    def intersects(self, ray):
        """Checks if ray intersects this sphere. Returns distance to intersection or None if there is no intersection"""
        rayorigin = ([ray.origin[0],ray.origin[1],ray.origin[2],0])
        S = np.matmul(self.matrix,ray.origin)
        S = S[:3]
        raydirection = np.array([ray.direction[0],ray.direction[1],ray.direction[2],0])
        Rd = np.matmul(self.matrix,raydirection)
        Rd = Rd[:3]
        a = np.dot(Rd, Rd)
        b = 2 * np.dot(Rd, S)
        c = np.dot(S,S) - 1
        discriminant = b * b - 4 * a * c

        if discriminant >= 0:
            dist1 = (-b - sqrt(discriminant)) / a 
            dist2 = (-b + sqrt(discriminant)) / a 
            #if dist1 > 0 and dist1 < dist2:
            if dist1 + 0.001 > 0: 
                return dist1
            elif dist2 + 0.001 > 0: 
                return dist2
        return None

class Light: 
    """implement lights
    LIGHT <name> <pos x> <pos y> <pos z> <Ir> <Ig> <Ib>""" 
    def __init__(self, lightorigin, lightcolour):
        self.origin = lightorigin
        self.colour = lightcolour

class Ray:
    """Ray is a point and a normalized direction""" 
    def  __init__(self, point, direction,depth):
        self.origin = point        
        self.direction = direction[:3]/np.linalg.norm(direction[:3])
        self.depth = depth

class Scene:
    """Scene has all the information needed for the ray tracing engine"""

    def __init__(self, eye, objects, width, height):
        self.eye = eye
        self.objects = objects
        self.width = width
        self.height = height

def save_imageP3(width, height, fname,passarray):
    
    with open(fname,'w') as f: 
        
        f.write("P3 {} {}\n255\n".format(width, height))
        passarray.reverse()
        for i in passarray: 
            for j in i: 
                for k in j: 
                    f.write("{} ".format(k))

def convert_to_world(ray, checklocaldist):
    rayorigin = ([ray.origin[0],ray.origin[1],ray.origin[2],1])
    raydirection = np.array([ray.direction[0],ray.direction[1],ray.direction[2],0])
    directionshift = raydirection * checklocaldist
    hit_intersection = rayorigin + directionshift 
    returndistance = sqrt(np.dot(hit_intersection - rayorigin,hit_intersection - rayorigin))
    return hit_intersection,returndistance

def raytrace(objects, ray, lights,ambience):
    backgrounddistance = 10000
    currentdistance = backgrounddistance
    hitobj = None
    hit_point = None
    sphere_point_normal = None


    for obj in objects: 
        
        checklocaldist = obj.intersects(ray)
        if checklocaldist is not None:
            hit_point_alpha, world_space_dist = convert_to_world(ray,checklocaldist)
            if world_space_dist < currentdistance: 
                currentdistance = world_space_dist
                hitobj = obj
                hit_point = hit_point_alpha
                inverse_transpose = obj.matrix.transpose()
                normal_matrix = np.matmul(inverse_transpose,obj.matrix)
                center = np.array([obj.translation[0], obj.translation[1] , obj.translation[2], 0])
                sphere_point_normal = np.matmul(normal_matrix,(hit_point - center))
                sphere_point_normal = sphere_point_normal[:3]
                sphere_point_normal = sphere_point_normal/np.linalg.norm(sphere_point_normal)
                sphere_coordinate_direction = ray.direction * checklocaldist
                sphere_coordinate_direction = np.array([sphere_coordinate_direction[0],sphere_coordinate_direction[1], sphere_coordinate_direction[2], 0])
                sphere_coordinate_hit_point = np.matmul(hitobj.matrix,ray.origin) + np.matmul(hitobj.matrix,sphere_coordinate_direction)        
                hit_point = sphere_coordinate_hit_point
    if hitobj is not None: 
        point_light_colourings = []
        for light in lights: 
            lightorigin = np.array([light.origin[0], light.origin[1], light.origin[2],1])
            light_blocked = False
            object_to_light = Ray(hit_point, lightorigin - hit_point, 1)

            for objs in objects: 
                check_if_blocked = objs.intersects(object_to_light)
                if check_if_blocked is not None: 
                    light_blocked = True
            #light_blocked = False
            if light_blocked == False: 
                #LIGHTS ARE ARRAY OF 3 WHILE OBJ.INTERSECTS TAKES ARRAY OF 4
                
                ray_to_light = Ray(hit_point, lightorigin - hit_point, 1)
                pointlightcolour = hitobj.diffuse * light.colour * max(np.dot(ray_to_light.direction[:3], sphere_point_normal),0) * hitobj.material 
                point_light_colourings.append(pointlightcolour)

                #Ks*Ip[c]*(R dot V)n 
                S = max(np.dot(ray_to_light.direction[:3], sphere_point_normal),0) * sphere_point_normal - ray_to_light.direction[:3]
                R = max(np.dot(ray_to_light.direction[:3], sphere_point_normal),0) * sphere_point_normal + S
                V = Ray(hit_point, ray.origin - hit_point,1)
                r = R/np.linalg.norm(R)
                v = V.direction[:3]/np.linalg.norm(V.direction[:3])

                pointlightcolour = hitobj.specular * light.colour * max(np.dot(r,v),0)**hitobj.shiny
                point_light_colourings.append(pointlightcolour)
            
        diffusespecular_light_sum = np.array([float(0),float(0),float(0)])
        for pcolour in point_light_colourings:
            diffusespecular_light_sum += pcolour            

        return hitobj.ambience * ambience * hitobj.material + diffusespecular_light_sum                      
 
    else: 
        return None

def render(scene, left, right, top, bottom,back,near,lightList,ambience,output):
    #set up the ray projection through the screen
    width = scene.width 
    height = scene.height 
    eye = scene.eye
    objects = scene.objects
    xscreen = right - left 
    yscreen = top - bottom 
    xstep = xscreen/(width - 1)
    ystep = yscreen/(height -1)
    lights = lightList
    image_array = [[None for _ in range(width)] for _ in range(height)]
    for i in range(height):

        y = bottom + i * ystep 

        for j in range(width):
            depth = 1
            x = left + j * xstep 
            # both np array distance from camera to pixel point 
            eyeray = Ray(eye, np.array([x,y,-near,1]) - eye, depth)
            raycolour = raytrace(objects, eyeray,lights,ambience)
            backgroundpixelcolour = bytecolour(back[0],back[1],back[2])
            if raycolour is not None: 
                
                raycolourbyte = bytecolour(raycolour[0],raycolour[1],raycolour[2])
                image_array[i][j] = (raycolourbyte[0],raycolourbyte[1],raycolourbyte[2])

            else:    
                image_array[i][j] = (backgroundpixelcolour[0],backgroundpixelcolour[1],backgroundpixelcolour[2])
    
    save_imageP3(width, height, output, image_array) 
#change colour to byte colour of 255 values
def bytecolour(red,green,blue): 
    red = min(round(red *255),255)
    green = min(round(green * 255),255)
    blue = min(round(blue * 255),255) 
    return [red,green,blue]    

def main():
    filename = sys.argv[1] 
    filecontent = [] 
    near = 0
    left = 0 
    right = 0 
    bottom = 0 
    top = 0
    sphereList = [] 
    lightList = [] 
    width = 0
    height = 0
    output = None 
    ambient = []
    back = []
    eye = np.array([0,0,0,1])

    with open(filename) as f: 
        filecontent= f.readlines()

    for line in filecontent: 
        if "NEAR" in line:
            splitline = line.split()
            near = int(splitline[1])
        if "LEFT" in line:
            splitline = line.split()
            left = int(splitline[1])
        if "RIGHT" in line:
            splitline = line.split()
            right = int(splitline[1])
        if "BOTTOM" in line:
            splitline = line.split()
            bottom = int(splitline[1]) 
        if "TOP" in line:
            splitline = line.split()
            top = int(splitline[1])
        if "SPHERE" in line:
            splitline = line.split()
            #(center, radius, material,scale,normal,lightproperties)
            #SPHERE <name> <pos x> <pos y> <pos z> <scl x> <scl y> <scl z> <r> <g> <b> <Ka> <Kd> <Ks> <Kr> <n>

            translation = np.array([float(splitline[2]),float(splitline[3]),float(splitline[4]),1])
            rad = 1
            scale = np.array([float(splitline[5]),float(splitline[6]),float(splitline[7])])
            shinyfactor = float(splitline[15])
            lightproperties = np.array([float(splitline[11]),float(splitline[12]),float(splitline[13]),float(splitline[14])])
            material = np.array([float(splitline[8]),float(splitline[9]),float(splitline[10])])
            insertSphere = Sphere(translation,rad,material,scale,shinyfactor,lightproperties)
            insertSphere.create_scale_matrix()
            sphereList.append(insertSphere)

        if "LIGHT" in line:
            #LIGHT <name> <pos x> <pos y> <pos z> <Ir> <Ig> <Ib>
            splitline = line.split()
            lightorigin = np.array([float(splitline[2]),float(splitline[3]),float(splitline[4])])
            lightcolour = np.array([float(splitline[5]),float(splitline[6]),float(splitline[7])])
            light = Light(lightorigin, lightcolour)
            lightList.append(light)


        if "RES" in line: 
            splitline = line.split()
            width = int(splitline[1])
            height = int(splitline[2])
        if "OUTPUT" in line: 
            splitline = line.split()
            output = splitline[1]
        if "AMBIENT" in line: 
            splitline = line.split()
            ambient = np.array([float(splitline[1]),float(splitline[2]),float(splitline[3])])
            
        if "BACK" in line: 
            splitline = line.split()
            backconvert = splitline[1:]
            for i in backconvert: 
                back.append(float(i))

    scene_to_make = Scene(eye, sphereList, width, height)
    render(scene_to_make, left, right, top, bottom,back,near,lightList,ambient,output)



if __name__ == "__main__":
    main()
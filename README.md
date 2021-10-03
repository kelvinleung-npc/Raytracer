# Raytracer
A basic ray tracer for class that handled lighting on ellipsoids. 
Takes text files of the form: 
NEAR <n>

LEFT <l>

RIGHT <r>

BOTTOM <b>

TOP <t>

RES <x> <y>

SPHERE <name> <pos x> <pos y> <pos z> <scl x> <scl y> <scl z> <r> <g> <b> <Ka> <Kd> <Ks> <Kr> <n>

… // up to 14 additional sphere specifications

LIGHT <name> <pos x> <pos y> <pos z> <Ir> <Ig> <Ib>

… // up to 9 additional light specifications

BACK <r> <g > <b>

AMBIENT <Ir> <Ig> <Ib>

OUTPUT <name>

and outputs a png file to be viewed 

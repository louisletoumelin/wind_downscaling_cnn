?? 
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
?
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
?
conv1_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1_0/kernel
y
"conv1_0/kernel/Read/ReadVariableOpReadVariableOpconv1_0/kernel*&
_output_shapes
: *
dtype0
p
conv1_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1_0/bias
i
 conv1_0/bias/Read/ReadVariableOpReadVariableOpconv1_0/bias*
_output_shapes
: *
dtype0
|
conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_nameconv1/kernel
u
 conv1/kernel/Read/ReadVariableOpReadVariableOpconv1/kernel*&
_output_shapes
:  *
dtype0
l

conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
conv1/bias
e
conv1/bias/Read/ReadVariableOpReadVariableOp
conv1/bias*
_output_shapes
: *
dtype0
?
conv2_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_nameconv2_0/kernel
y
"conv2_0/kernel/Read/ReadVariableOpReadVariableOpconv2_0/kernel*&
_output_shapes
: @*
dtype0
p
conv2_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2_0/bias
i
 conv2_0/bias/Read/ReadVariableOpReadVariableOpconv2_0/bias*
_output_shapes
:@*
dtype0
|
conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_nameconv2/kernel
u
 conv2/kernel/Read/ReadVariableOpReadVariableOpconv2/kernel*&
_output_shapes
:@@*
dtype0
l

conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
conv2/bias
e
conv2/bias/Read/ReadVariableOpReadVariableOp
conv2/bias*
_output_shapes
:@*
dtype0
?
conv3_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*
shared_nameconv3_0/kernel
z
"conv3_0/kernel/Read/ReadVariableOpReadVariableOpconv3_0/kernel*'
_output_shapes
:@?*
dtype0
q
conv3_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv3_0/bias
j
 conv3_0/bias/Read/ReadVariableOpReadVariableOpconv3_0/bias*
_output_shapes	
:?*
dtype0
~
conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameconv3/kernel
w
 conv3/kernel/Read/ReadVariableOpReadVariableOpconv3/kernel*(
_output_shapes
:??*
dtype0
m

conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
conv3/bias
f
conv3/bias/Read/ReadVariableOpReadVariableOp
conv3/bias*
_output_shapes	
:?*
dtype0
?
conv4_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameconv4_0/kernel
{
"conv4_0/kernel/Read/ReadVariableOpReadVariableOpconv4_0/kernel*(
_output_shapes
:??*
dtype0
q
conv4_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv4_0/bias
j
 conv4_0/bias/Read/ReadVariableOpReadVariableOpconv4_0/bias*
_output_shapes	
:?*
dtype0
~
conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameconv4/kernel
w
 conv4/kernel/Read/ReadVariableOpReadVariableOpconv4/kernel*(
_output_shapes
:??*
dtype0
m

conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
conv4/bias
f
conv4/bias/Read/ReadVariableOpReadVariableOp
conv4/bias*
_output_shapes	
:?*
dtype0
z

up4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_name
up4/kernel
s
up4/kernel/Read/ReadVariableOpReadVariableOp
up4/kernel*(
_output_shapes
:??*
dtype0
i
up4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
up4/bias
b
up4/bias/Read/ReadVariableOpReadVariableOpup4/bias*
_output_shapes	
:?*
dtype0
?
conv3_up_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*"
shared_nameconv3_up_0/kernel
?
%conv3_up_0/kernel/Read/ReadVariableOpReadVariableOpconv3_up_0/kernel*(
_output_shapes
:??*
dtype0
w
conv3_up_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_nameconv3_up_0/bias
p
#conv3_up_0/bias/Read/ReadVariableOpReadVariableOpconv3_up_0/bias*
_output_shapes	
:?*
dtype0
?
conv3_up/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??* 
shared_nameconv3_up/kernel
}
#conv3_up/kernel/Read/ReadVariableOpReadVariableOpconv3_up/kernel*(
_output_shapes
:??*
dtype0
s
conv3_up/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv3_up/bias
l
!conv3_up/bias/Read/ReadVariableOpReadVariableOpconv3_up/bias*
_output_shapes	
:?*
dtype0
y

up3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*
shared_name
up3/kernel
r
up3/kernel/Read/ReadVariableOpReadVariableOp
up3/kernel*'
_output_shapes
:?@*
dtype0
h
up3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
up3/bias
a
up3/bias/Read/ReadVariableOpReadVariableOpup3/bias*
_output_shapes
:@*
dtype0
?
conv2_up_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*"
shared_nameconv2_up_0/kernel
?
%conv2_up_0/kernel/Read/ReadVariableOpReadVariableOpconv2_up_0/kernel*'
_output_shapes
:?@*
dtype0
v
conv2_up_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2_up_0/bias
o
#conv2_up_0/bias/Read/ReadVariableOpReadVariableOpconv2_up_0/bias*
_output_shapes
:@*
dtype0
?
conv2_up/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2_up/kernel
{
#conv2_up/kernel/Read/ReadVariableOpReadVariableOpconv2_up/kernel*&
_output_shapes
:@@*
dtype0
r
conv2_up/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2_up/bias
k
!conv2_up/bias/Read/ReadVariableOpReadVariableOpconv2_up/bias*
_output_shapes
:@*
dtype0
x

up2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *
shared_name
up2/kernel
q
up2/kernel/Read/ReadVariableOpReadVariableOp
up2/kernel*&
_output_shapes
:@ *
dtype0
h
up2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
up2/bias
a
up2/bias/Read/ReadVariableOpReadVariableOpup2/bias*
_output_shapes
: *
dtype0
?
conv1_up_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *"
shared_nameconv1_up_0/kernel

%conv1_up_0/kernel/Read/ReadVariableOpReadVariableOpconv1_up_0/kernel*&
_output_shapes
:@ *
dtype0
v
conv1_up_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv1_up_0/bias
o
#conv1_up_0/bias/Read/ReadVariableOpReadVariableOpconv1_up_0/bias*
_output_shapes
: *
dtype0
?
conv1_up/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv1_up/kernel
{
#conv1_up/kernel/Read/ReadVariableOpReadVariableOpconv1_up/kernel*&
_output_shapes
:  *
dtype0
r
conv1_up/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1_up/bias
k
!conv1_up/bias/Read/ReadVariableOpReadVariableOpconv1_up/bias*
_output_shapes
: *
dtype0
?
conv1_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1_1/kernel
y
"conv1_1/kernel/Read/ReadVariableOpReadVariableOpconv1_1/kernel*&
_output_shapes
: *
dtype0
p
conv1_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1_1/bias
i
 conv1_1/bias/Read/ReadVariableOpReadVariableOpconv1_1/bias*
_output_shapes
:*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
?
RMSprop/conv1_0/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameRMSprop/conv1_0/kernel/rms
?
.RMSprop/conv1_0/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv1_0/kernel/rms*&
_output_shapes
: *
dtype0
?
RMSprop/conv1_0/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameRMSprop/conv1_0/bias/rms
?
,RMSprop/conv1_0/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv1_0/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/conv1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *)
shared_nameRMSprop/conv1/kernel/rms
?
,RMSprop/conv1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv1/kernel/rms*&
_output_shapes
:  *
dtype0
?
RMSprop/conv1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameRMSprop/conv1/bias/rms
}
*RMSprop/conv1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv1/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/conv2_0/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameRMSprop/conv2_0/kernel/rms
?
.RMSprop/conv2_0/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2_0/kernel/rms*&
_output_shapes
: @*
dtype0
?
RMSprop/conv2_0/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameRMSprop/conv2_0/bias/rms
?
,RMSprop/conv2_0/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2_0/bias/rms*
_output_shapes
:@*
dtype0
?
RMSprop/conv2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*)
shared_nameRMSprop/conv2/kernel/rms
?
,RMSprop/conv2/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2/kernel/rms*&
_output_shapes
:@@*
dtype0
?
RMSprop/conv2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameRMSprop/conv2/bias/rms
}
*RMSprop/conv2/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2/bias/rms*
_output_shapes
:@*
dtype0
?
RMSprop/conv3_0/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*+
shared_nameRMSprop/conv3_0/kernel/rms
?
.RMSprop/conv3_0/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv3_0/kernel/rms*'
_output_shapes
:@?*
dtype0
?
RMSprop/conv3_0/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameRMSprop/conv3_0/bias/rms
?
,RMSprop/conv3_0/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv3_0/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/conv3/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*)
shared_nameRMSprop/conv3/kernel/rms
?
,RMSprop/conv3/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv3/kernel/rms*(
_output_shapes
:??*
dtype0
?
RMSprop/conv3/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameRMSprop/conv3/bias/rms
~
*RMSprop/conv3/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv3/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/conv4_0/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*+
shared_nameRMSprop/conv4_0/kernel/rms
?
.RMSprop/conv4_0/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv4_0/kernel/rms*(
_output_shapes
:??*
dtype0
?
RMSprop/conv4_0/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameRMSprop/conv4_0/bias/rms
?
,RMSprop/conv4_0/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv4_0/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/conv4/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*)
shared_nameRMSprop/conv4/kernel/rms
?
,RMSprop/conv4/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv4/kernel/rms*(
_output_shapes
:??*
dtype0
?
RMSprop/conv4/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameRMSprop/conv4/bias/rms
~
*RMSprop/conv4/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv4/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/up4/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*'
shared_nameRMSprop/up4/kernel/rms
?
*RMSprop/up4/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/up4/kernel/rms*(
_output_shapes
:??*
dtype0
?
RMSprop/up4/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameRMSprop/up4/bias/rms
z
(RMSprop/up4/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/up4/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/conv3_up_0/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*.
shared_nameRMSprop/conv3_up_0/kernel/rms
?
1RMSprop/conv3_up_0/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv3_up_0/kernel/rms*(
_output_shapes
:??*
dtype0
?
RMSprop/conv3_up_0/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_nameRMSprop/conv3_up_0/bias/rms
?
/RMSprop/conv3_up_0/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv3_up_0/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/conv3_up/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*,
shared_nameRMSprop/conv3_up/kernel/rms
?
/RMSprop/conv3_up/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv3_up/kernel/rms*(
_output_shapes
:??*
dtype0
?
RMSprop/conv3_up/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_nameRMSprop/conv3_up/bias/rms
?
-RMSprop/conv3_up/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv3_up/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/up3/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*'
shared_nameRMSprop/up3/kernel/rms
?
*RMSprop/up3/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/up3/kernel/rms*'
_output_shapes
:?@*
dtype0
?
RMSprop/up3/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameRMSprop/up3/bias/rms
y
(RMSprop/up3/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/up3/bias/rms*
_output_shapes
:@*
dtype0
?
RMSprop/conv2_up_0/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*.
shared_nameRMSprop/conv2_up_0/kernel/rms
?
1RMSprop/conv2_up_0/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2_up_0/kernel/rms*'
_output_shapes
:?@*
dtype0
?
RMSprop/conv2_up_0/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameRMSprop/conv2_up_0/bias/rms
?
/RMSprop/conv2_up_0/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2_up_0/bias/rms*
_output_shapes
:@*
dtype0
?
RMSprop/conv2_up/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*,
shared_nameRMSprop/conv2_up/kernel/rms
?
/RMSprop/conv2_up/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2_up/kernel/rms*&
_output_shapes
:@@*
dtype0
?
RMSprop/conv2_up/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameRMSprop/conv2_up/bias/rms
?
-RMSprop/conv2_up/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2_up/bias/rms*
_output_shapes
:@*
dtype0
?
RMSprop/up2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *'
shared_nameRMSprop/up2/kernel/rms
?
*RMSprop/up2/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/up2/kernel/rms*&
_output_shapes
:@ *
dtype0
?
RMSprop/up2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameRMSprop/up2/bias/rms
y
(RMSprop/up2/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/up2/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/conv1_up_0/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *.
shared_nameRMSprop/conv1_up_0/kernel/rms
?
1RMSprop/conv1_up_0/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv1_up_0/kernel/rms*&
_output_shapes
:@ *
dtype0
?
RMSprop/conv1_up_0/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameRMSprop/conv1_up_0/bias/rms
?
/RMSprop/conv1_up_0/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv1_up_0/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/conv1_up/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *,
shared_nameRMSprop/conv1_up/kernel/rms
?
/RMSprop/conv1_up/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv1_up/kernel/rms*&
_output_shapes
:  *
dtype0
?
RMSprop/conv1_up/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameRMSprop/conv1_up/bias/rms
?
-RMSprop/conv1_up/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv1_up/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/conv1_1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameRMSprop/conv1_1/kernel/rms
?
.RMSprop/conv1_1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv1_1/kernel/rms*&
_output_shapes
: *
dtype0
?
RMSprop/conv1_1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/conv1_1/bias/rms
?
,RMSprop/conv1_1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv1_1/bias/rms*
_output_shapes
:*
dtype0

NoOpNoOp
֮
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer-14
layer-15
layer_with_weights-8
layer-16
layer-17
layer-18
layer_with_weights-9
layer-19
layer_with_weights-10
layer-20
layer-21
layer_with_weights-11
layer-22
layer-23
layer-24
layer_with_weights-12
layer-25
layer_with_weights-13
layer-26
layer-27
layer_with_weights-14
layer-28
layer-29
layer_with_weights-15
layer-30
 layer_with_weights-16
 layer-31
!layer_with_weights-17
!layer-32
"layer-33
#	optimizer
$regularization_losses
%trainable_variables
&	variables
'	keras_api
(
signatures
 
R
)regularization_losses
*trainable_variables
+	variables
,	keras_api
h

-kernel
.bias
/regularization_losses
0trainable_variables
1	variables
2	keras_api
h

3kernel
4bias
5regularization_losses
6trainable_variables
7	variables
8	keras_api
R
9regularization_losses
:trainable_variables
;	variables
<	keras_api
h

=kernel
>bias
?regularization_losses
@trainable_variables
A	variables
B	keras_api
h

Ckernel
Dbias
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
R
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
h

Mkernel
Nbias
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
h

Skernel
Tbias
Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
R
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
R
]regularization_losses
^trainable_variables
_	variables
`	keras_api
h

akernel
bbias
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
h

gkernel
hbias
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
R
mregularization_losses
ntrainable_variables
o	variables
p	keras_api
R
qregularization_losses
rtrainable_variables
s	variables
t	keras_api
h

ukernel
vbias
wregularization_losses
xtrainable_variables
y	variables
z	keras_api
R
{regularization_losses
|trainable_variables
}	variables
~	keras_api
U
regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?iter

?decay
?learning_rate
?momentum
?rho
-rms?
.rms?
3rms?
4rms?
=rms?
>rms?
Crms?
Drms?
Mrms?
Nrms?
Srms?
Trms?
arms?
brms?
grms?
hrms?
urms?
vrms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms?
 
?
-0
.1
32
43
=4
>5
C6
D7
M8
N9
S10
T11
a12
b13
g14
h15
u16
v17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?
-0
.1
32
43
=4
>5
C6
D7
M8
N9
S10
T11
a12
b13
g14
h15
u16
v17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?
$regularization_losses
%trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
&	variables
?layer_metrics
 
 
 
 
?
)regularization_losses
 ?layer_regularization_losses
*trainable_variables
?non_trainable_variables
?layers
?metrics
+	variables
?layer_metrics
ZX
VARIABLE_VALUEconv1_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv1_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

-0
.1

-0
.1
?
/regularization_losses
 ?layer_regularization_losses
0trainable_variables
?non_trainable_variables
?layers
?metrics
1	variables
?layer_metrics
XV
VARIABLE_VALUEconv1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

30
41

30
41
?
5regularization_losses
 ?layer_regularization_losses
6trainable_variables
?non_trainable_variables
?layers
?metrics
7	variables
?layer_metrics
 
 
 
?
9regularization_losses
 ?layer_regularization_losses
:trainable_variables
?non_trainable_variables
?layers
?metrics
;	variables
?layer_metrics
ZX
VARIABLE_VALUEconv2_0/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2_0/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

=0
>1

=0
>1
?
?regularization_losses
 ?layer_regularization_losses
@trainable_variables
?non_trainable_variables
?layers
?metrics
A	variables
?layer_metrics
XV
VARIABLE_VALUEconv2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

C0
D1

C0
D1
?
Eregularization_losses
 ?layer_regularization_losses
Ftrainable_variables
?non_trainable_variables
?layers
?metrics
G	variables
?layer_metrics
 
 
 
?
Iregularization_losses
 ?layer_regularization_losses
Jtrainable_variables
?non_trainable_variables
?layers
?metrics
K	variables
?layer_metrics
ZX
VARIABLE_VALUEconv3_0/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv3_0/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

M0
N1

M0
N1
?
Oregularization_losses
 ?layer_regularization_losses
Ptrainable_variables
?non_trainable_variables
?layers
?metrics
Q	variables
?layer_metrics
XV
VARIABLE_VALUEconv3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

S0
T1

S0
T1
?
Uregularization_losses
 ?layer_regularization_losses
Vtrainable_variables
?non_trainable_variables
?layers
?metrics
W	variables
?layer_metrics
 
 
 
?
Yregularization_losses
 ?layer_regularization_losses
Ztrainable_variables
?non_trainable_variables
?layers
?metrics
[	variables
?layer_metrics
 
 
 
?
]regularization_losses
 ?layer_regularization_losses
^trainable_variables
?non_trainable_variables
?layers
?metrics
_	variables
?layer_metrics
ZX
VARIABLE_VALUEconv4_0/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv4_0/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

a0
b1

a0
b1
?
cregularization_losses
 ?layer_regularization_losses
dtrainable_variables
?non_trainable_variables
?layers
?metrics
e	variables
?layer_metrics
XV
VARIABLE_VALUEconv4/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv4/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

g0
h1

g0
h1
?
iregularization_losses
 ?layer_regularization_losses
jtrainable_variables
?non_trainable_variables
?layers
?metrics
k	variables
?layer_metrics
 
 
 
?
mregularization_losses
 ?layer_regularization_losses
ntrainable_variables
?non_trainable_variables
?layers
?metrics
o	variables
?layer_metrics
 
 
 
?
qregularization_losses
 ?layer_regularization_losses
rtrainable_variables
?non_trainable_variables
?layers
?metrics
s	variables
?layer_metrics
VT
VARIABLE_VALUE
up4/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEup4/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

u0
v1

u0
v1
?
wregularization_losses
 ?layer_regularization_losses
xtrainable_variables
?non_trainable_variables
?layers
?metrics
y	variables
?layer_metrics
 
 
 
?
{regularization_losses
 ?layer_regularization_losses
|trainable_variables
?non_trainable_variables
?layers
?metrics
}	variables
?layer_metrics
 
 
 
?
regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
][
VARIABLE_VALUEconv3_up_0/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3_up_0/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
\Z
VARIABLE_VALUEconv3_up/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv3_up/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
 
 
 
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
WU
VARIABLE_VALUE
up3/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEup3/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
 
 
 
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
 
 
 
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
^\
VARIABLE_VALUEconv2_up_0/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv2_up_0/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
\Z
VARIABLE_VALUEconv2_up/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2_up/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
 
 
 
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
WU
VARIABLE_VALUE
up2/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEup2/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
 
 
 
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
^\
VARIABLE_VALUEconv1_up_0/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv1_up_0/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
\Z
VARIABLE_VALUEconv1_up/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1_up/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
[Y
VARIABLE_VALUEconv1_1/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1_1/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
 
 
 
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
 
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33

?0
?1
?2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUERMSprop/conv1_0/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUERMSprop/conv1_0/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv1/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUERMSprop/conv1/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2_0/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUERMSprop/conv2_0/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUERMSprop/conv2/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv3_0/kernel/rmsTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUERMSprop/conv3_0/bias/rmsRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv3/kernel/rmsTlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUERMSprop/conv3/bias/rmsRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv4_0/kernel/rmsTlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUERMSprop/conv4_0/bias/rmsRlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv4/kernel/rmsTlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUERMSprop/conv4/bias/rmsRlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUERMSprop/up4/kernel/rmsTlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUERMSprop/up4/bias/rmsRlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv3_up_0/kernel/rmsTlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv3_up_0/bias/rmsRlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv3_up/kernel/rmsUlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv3_up/bias/rmsSlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUERMSprop/up3/kernel/rmsUlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUERMSprop/up3/bias/rmsSlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2_up_0/kernel/rmsUlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2_up_0/bias/rmsSlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2_up/kernel/rmsUlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2_up/bias/rmsSlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUERMSprop/up2/kernel/rmsUlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUERMSprop/up2/bias/rmsSlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv1_up_0/kernel/rmsUlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv1_up_0/bias/rmsSlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv1_up/kernel/rmsUlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv1_up/bias/rmsSlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv1_1/kernel/rmsUlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUERMSprop/conv1_1/bias/rmsSlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????OE*
dtype0*$
shape:?????????OE
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv1_0/kernelconv1_0/biasconv1/kernel
conv1/biasconv2_0/kernelconv2_0/biasconv2/kernel
conv2/biasconv3_0/kernelconv3_0/biasconv3/kernel
conv3/biasconv4_0/kernelconv4_0/biasconv4/kernel
conv4/bias
up4/kernelup4/biasconv3_up_0/kernelconv3_up_0/biasconv3_up/kernelconv3_up/bias
up3/kernelup3/biasconv2_up_0/kernelconv2_up_0/biasconv2_up/kernelconv2_up/bias
up2/kernelup2/biasconv1_up_0/kernelconv1_up_0/biasconv1_up/kernelconv1_up/biasconv1_1/kernelconv1_1/bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????OE*F
_read_only_resource_inputs(
&$	
 !"#$*2
config_proto" 

CPU

GPU2 *0J 8? *-
f(R&
$__inference_signature_wrapper_204454
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"conv1_0/kernel/Read/ReadVariableOp conv1_0/bias/Read/ReadVariableOp conv1/kernel/Read/ReadVariableOpconv1/bias/Read/ReadVariableOp"conv2_0/kernel/Read/ReadVariableOp conv2_0/bias/Read/ReadVariableOp conv2/kernel/Read/ReadVariableOpconv2/bias/Read/ReadVariableOp"conv3_0/kernel/Read/ReadVariableOp conv3_0/bias/Read/ReadVariableOp conv3/kernel/Read/ReadVariableOpconv3/bias/Read/ReadVariableOp"conv4_0/kernel/Read/ReadVariableOp conv4_0/bias/Read/ReadVariableOp conv4/kernel/Read/ReadVariableOpconv4/bias/Read/ReadVariableOpup4/kernel/Read/ReadVariableOpup4/bias/Read/ReadVariableOp%conv3_up_0/kernel/Read/ReadVariableOp#conv3_up_0/bias/Read/ReadVariableOp#conv3_up/kernel/Read/ReadVariableOp!conv3_up/bias/Read/ReadVariableOpup3/kernel/Read/ReadVariableOpup3/bias/Read/ReadVariableOp%conv2_up_0/kernel/Read/ReadVariableOp#conv2_up_0/bias/Read/ReadVariableOp#conv2_up/kernel/Read/ReadVariableOp!conv2_up/bias/Read/ReadVariableOpup2/kernel/Read/ReadVariableOpup2/bias/Read/ReadVariableOp%conv1_up_0/kernel/Read/ReadVariableOp#conv1_up_0/bias/Read/ReadVariableOp#conv1_up/kernel/Read/ReadVariableOp!conv1_up/bias/Read/ReadVariableOp"conv1_1/kernel/Read/ReadVariableOp conv1_1/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp.RMSprop/conv1_0/kernel/rms/Read/ReadVariableOp,RMSprop/conv1_0/bias/rms/Read/ReadVariableOp,RMSprop/conv1/kernel/rms/Read/ReadVariableOp*RMSprop/conv1/bias/rms/Read/ReadVariableOp.RMSprop/conv2_0/kernel/rms/Read/ReadVariableOp,RMSprop/conv2_0/bias/rms/Read/ReadVariableOp,RMSprop/conv2/kernel/rms/Read/ReadVariableOp*RMSprop/conv2/bias/rms/Read/ReadVariableOp.RMSprop/conv3_0/kernel/rms/Read/ReadVariableOp,RMSprop/conv3_0/bias/rms/Read/ReadVariableOp,RMSprop/conv3/kernel/rms/Read/ReadVariableOp*RMSprop/conv3/bias/rms/Read/ReadVariableOp.RMSprop/conv4_0/kernel/rms/Read/ReadVariableOp,RMSprop/conv4_0/bias/rms/Read/ReadVariableOp,RMSprop/conv4/kernel/rms/Read/ReadVariableOp*RMSprop/conv4/bias/rms/Read/ReadVariableOp*RMSprop/up4/kernel/rms/Read/ReadVariableOp(RMSprop/up4/bias/rms/Read/ReadVariableOp1RMSprop/conv3_up_0/kernel/rms/Read/ReadVariableOp/RMSprop/conv3_up_0/bias/rms/Read/ReadVariableOp/RMSprop/conv3_up/kernel/rms/Read/ReadVariableOp-RMSprop/conv3_up/bias/rms/Read/ReadVariableOp*RMSprop/up3/kernel/rms/Read/ReadVariableOp(RMSprop/up3/bias/rms/Read/ReadVariableOp1RMSprop/conv2_up_0/kernel/rms/Read/ReadVariableOp/RMSprop/conv2_up_0/bias/rms/Read/ReadVariableOp/RMSprop/conv2_up/kernel/rms/Read/ReadVariableOp-RMSprop/conv2_up/bias/rms/Read/ReadVariableOp*RMSprop/up2/kernel/rms/Read/ReadVariableOp(RMSprop/up2/bias/rms/Read/ReadVariableOp1RMSprop/conv1_up_0/kernel/rms/Read/ReadVariableOp/RMSprop/conv1_up_0/bias/rms/Read/ReadVariableOp/RMSprop/conv1_up/kernel/rms/Read/ReadVariableOp-RMSprop/conv1_up/bias/rms/Read/ReadVariableOp.RMSprop/conv1_1/kernel/rms/Read/ReadVariableOp,RMSprop/conv1_1/bias/rms/Read/ReadVariableOpConst*`
TinY
W2U	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *(
f#R!
__inference__traced_save_205670
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1_0/kernelconv1_0/biasconv1/kernel
conv1/biasconv2_0/kernelconv2_0/biasconv2/kernel
conv2/biasconv3_0/kernelconv3_0/biasconv3/kernel
conv3/biasconv4_0/kernelconv4_0/biasconv4/kernel
conv4/bias
up4/kernelup4/biasconv3_up_0/kernelconv3_up_0/biasconv3_up/kernelconv3_up/bias
up3/kernelup3/biasconv2_up_0/kernelconv2_up_0/biasconv2_up/kernelconv2_up/bias
up2/kernelup2/biasconv1_up_0/kernelconv1_up_0/biasconv1_up/kernelconv1_up/biasconv1_1/kernelconv1_1/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcounttotal_1count_1total_2count_2RMSprop/conv1_0/kernel/rmsRMSprop/conv1_0/bias/rmsRMSprop/conv1/kernel/rmsRMSprop/conv1/bias/rmsRMSprop/conv2_0/kernel/rmsRMSprop/conv2_0/bias/rmsRMSprop/conv2/kernel/rmsRMSprop/conv2/bias/rmsRMSprop/conv3_0/kernel/rmsRMSprop/conv3_0/bias/rmsRMSprop/conv3/kernel/rmsRMSprop/conv3/bias/rmsRMSprop/conv4_0/kernel/rmsRMSprop/conv4_0/bias/rmsRMSprop/conv4/kernel/rmsRMSprop/conv4/bias/rmsRMSprop/up4/kernel/rmsRMSprop/up4/bias/rmsRMSprop/conv3_up_0/kernel/rmsRMSprop/conv3_up_0/bias/rmsRMSprop/conv3_up/kernel/rmsRMSprop/conv3_up/bias/rmsRMSprop/up3/kernel/rmsRMSprop/up3/bias/rmsRMSprop/conv2_up_0/kernel/rmsRMSprop/conv2_up_0/bias/rmsRMSprop/conv2_up/kernel/rmsRMSprop/conv2_up/bias/rmsRMSprop/up2/kernel/rmsRMSprop/up2/bias/rmsRMSprop/conv1_up_0/kernel/rmsRMSprop/conv1_up_0/bias/rmsRMSprop/conv1_up/kernel/rmsRMSprop/conv1_up/bias/rmsRMSprop/conv1_1/kernel/rmsRMSprop/conv1_1/bias/rms*_
TinX
V2T*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__traced_restore_205929æ
?	
b
F__inference_cropping2d_layer_call_and_return_conditional_losses_203119

inputs
identity?
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"    ????????    2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*J
_output_shapes8
6:4????????????????????????????????????*

begin_mask	*
end_mask	2
strided_slice?
IdentityIdentitystrided_slice:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
p
D__inference_concat_3_layer_call_and_return_conditional_losses_205193
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:??????????:,????????????????????????????:Z V
0
_output_shapes
:??????????
"
_user_specified_name
inputs/0:lh
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/1
?
?
&__inference_conv1_layer_call_fn_204975

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_2031612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????PF 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????PF : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????PF 
 
_user_specified_nameinputs
?
?
D__inference_conv1_up_layer_call_and_return_conditional_losses_203465

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????PF 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????PF 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????PF : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????PF 
 
_user_specified_nameinputs
?
U
)__inference_concat_2_layer_call_fn_205259
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????(#?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_concat_2_layer_call_and_return_conditional_losses_2033742
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????(#?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:?????????(#@:+???????????????????????????@:Y U
/
_output_shapes
:?????????(#@
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+???????????????????????????@
"
_user_specified_name
inputs/1
?
?
&__inference_conv4_layer_call_fn_205122

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv4_layer_call_and_return_conditional_losses_2032732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????
?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????
?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
?
 
_user_specified_nameinputs
??
?!
__inference__traced_save_205670
file_prefix-
)savev2_conv1_0_kernel_read_readvariableop+
'savev2_conv1_0_bias_read_readvariableop+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop-
)savev2_conv2_0_kernel_read_readvariableop+
'savev2_conv2_0_bias_read_readvariableop+
'savev2_conv2_kernel_read_readvariableop)
%savev2_conv2_bias_read_readvariableop-
)savev2_conv3_0_kernel_read_readvariableop+
'savev2_conv3_0_bias_read_readvariableop+
'savev2_conv3_kernel_read_readvariableop)
%savev2_conv3_bias_read_readvariableop-
)savev2_conv4_0_kernel_read_readvariableop+
'savev2_conv4_0_bias_read_readvariableop+
'savev2_conv4_kernel_read_readvariableop)
%savev2_conv4_bias_read_readvariableop)
%savev2_up4_kernel_read_readvariableop'
#savev2_up4_bias_read_readvariableop0
,savev2_conv3_up_0_kernel_read_readvariableop.
*savev2_conv3_up_0_bias_read_readvariableop.
*savev2_conv3_up_kernel_read_readvariableop,
(savev2_conv3_up_bias_read_readvariableop)
%savev2_up3_kernel_read_readvariableop'
#savev2_up3_bias_read_readvariableop0
,savev2_conv2_up_0_kernel_read_readvariableop.
*savev2_conv2_up_0_bias_read_readvariableop.
*savev2_conv2_up_kernel_read_readvariableop,
(savev2_conv2_up_bias_read_readvariableop)
%savev2_up2_kernel_read_readvariableop'
#savev2_up2_bias_read_readvariableop0
,savev2_conv1_up_0_kernel_read_readvariableop.
*savev2_conv1_up_0_bias_read_readvariableop.
*savev2_conv1_up_kernel_read_readvariableop,
(savev2_conv1_up_bias_read_readvariableop-
)savev2_conv1_1_kernel_read_readvariableop+
'savev2_conv1_1_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop9
5savev2_rmsprop_conv1_0_kernel_rms_read_readvariableop7
3savev2_rmsprop_conv1_0_bias_rms_read_readvariableop7
3savev2_rmsprop_conv1_kernel_rms_read_readvariableop5
1savev2_rmsprop_conv1_bias_rms_read_readvariableop9
5savev2_rmsprop_conv2_0_kernel_rms_read_readvariableop7
3savev2_rmsprop_conv2_0_bias_rms_read_readvariableop7
3savev2_rmsprop_conv2_kernel_rms_read_readvariableop5
1savev2_rmsprop_conv2_bias_rms_read_readvariableop9
5savev2_rmsprop_conv3_0_kernel_rms_read_readvariableop7
3savev2_rmsprop_conv3_0_bias_rms_read_readvariableop7
3savev2_rmsprop_conv3_kernel_rms_read_readvariableop5
1savev2_rmsprop_conv3_bias_rms_read_readvariableop9
5savev2_rmsprop_conv4_0_kernel_rms_read_readvariableop7
3savev2_rmsprop_conv4_0_bias_rms_read_readvariableop7
3savev2_rmsprop_conv4_kernel_rms_read_readvariableop5
1savev2_rmsprop_conv4_bias_rms_read_readvariableop5
1savev2_rmsprop_up4_kernel_rms_read_readvariableop3
/savev2_rmsprop_up4_bias_rms_read_readvariableop<
8savev2_rmsprop_conv3_up_0_kernel_rms_read_readvariableop:
6savev2_rmsprop_conv3_up_0_bias_rms_read_readvariableop:
6savev2_rmsprop_conv3_up_kernel_rms_read_readvariableop8
4savev2_rmsprop_conv3_up_bias_rms_read_readvariableop5
1savev2_rmsprop_up3_kernel_rms_read_readvariableop3
/savev2_rmsprop_up3_bias_rms_read_readvariableop<
8savev2_rmsprop_conv2_up_0_kernel_rms_read_readvariableop:
6savev2_rmsprop_conv2_up_0_bias_rms_read_readvariableop:
6savev2_rmsprop_conv2_up_kernel_rms_read_readvariableop8
4savev2_rmsprop_conv2_up_bias_rms_read_readvariableop5
1savev2_rmsprop_up2_kernel_rms_read_readvariableop3
/savev2_rmsprop_up2_bias_rms_read_readvariableop<
8savev2_rmsprop_conv1_up_0_kernel_rms_read_readvariableop:
6savev2_rmsprop_conv1_up_0_bias_rms_read_readvariableop:
6savev2_rmsprop_conv1_up_kernel_rms_read_readvariableop8
4savev2_rmsprop_conv1_up_bias_rms_read_readvariableop9
5savev2_rmsprop_conv1_1_kernel_rms_read_readvariableop7
3savev2_rmsprop_conv1_1_bias_rms_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?-
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*?,
value?,B?,TB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*?
value?B?TB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices? 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_conv1_0_kernel_read_readvariableop'savev2_conv1_0_bias_read_readvariableop'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop)savev2_conv2_0_kernel_read_readvariableop'savev2_conv2_0_bias_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop)savev2_conv3_0_kernel_read_readvariableop'savev2_conv3_0_bias_read_readvariableop'savev2_conv3_kernel_read_readvariableop%savev2_conv3_bias_read_readvariableop)savev2_conv4_0_kernel_read_readvariableop'savev2_conv4_0_bias_read_readvariableop'savev2_conv4_kernel_read_readvariableop%savev2_conv4_bias_read_readvariableop%savev2_up4_kernel_read_readvariableop#savev2_up4_bias_read_readvariableop,savev2_conv3_up_0_kernel_read_readvariableop*savev2_conv3_up_0_bias_read_readvariableop*savev2_conv3_up_kernel_read_readvariableop(savev2_conv3_up_bias_read_readvariableop%savev2_up3_kernel_read_readvariableop#savev2_up3_bias_read_readvariableop,savev2_conv2_up_0_kernel_read_readvariableop*savev2_conv2_up_0_bias_read_readvariableop*savev2_conv2_up_kernel_read_readvariableop(savev2_conv2_up_bias_read_readvariableop%savev2_up2_kernel_read_readvariableop#savev2_up2_bias_read_readvariableop,savev2_conv1_up_0_kernel_read_readvariableop*savev2_conv1_up_0_bias_read_readvariableop*savev2_conv1_up_kernel_read_readvariableop(savev2_conv1_up_bias_read_readvariableop)savev2_conv1_1_kernel_read_readvariableop'savev2_conv1_1_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop5savev2_rmsprop_conv1_0_kernel_rms_read_readvariableop3savev2_rmsprop_conv1_0_bias_rms_read_readvariableop3savev2_rmsprop_conv1_kernel_rms_read_readvariableop1savev2_rmsprop_conv1_bias_rms_read_readvariableop5savev2_rmsprop_conv2_0_kernel_rms_read_readvariableop3savev2_rmsprop_conv2_0_bias_rms_read_readvariableop3savev2_rmsprop_conv2_kernel_rms_read_readvariableop1savev2_rmsprop_conv2_bias_rms_read_readvariableop5savev2_rmsprop_conv3_0_kernel_rms_read_readvariableop3savev2_rmsprop_conv3_0_bias_rms_read_readvariableop3savev2_rmsprop_conv3_kernel_rms_read_readvariableop1savev2_rmsprop_conv3_bias_rms_read_readvariableop5savev2_rmsprop_conv4_0_kernel_rms_read_readvariableop3savev2_rmsprop_conv4_0_bias_rms_read_readvariableop3savev2_rmsprop_conv4_kernel_rms_read_readvariableop1savev2_rmsprop_conv4_bias_rms_read_readvariableop1savev2_rmsprop_up4_kernel_rms_read_readvariableop/savev2_rmsprop_up4_bias_rms_read_readvariableop8savev2_rmsprop_conv3_up_0_kernel_rms_read_readvariableop6savev2_rmsprop_conv3_up_0_bias_rms_read_readvariableop6savev2_rmsprop_conv3_up_kernel_rms_read_readvariableop4savev2_rmsprop_conv3_up_bias_rms_read_readvariableop1savev2_rmsprop_up3_kernel_rms_read_readvariableop/savev2_rmsprop_up3_bias_rms_read_readvariableop8savev2_rmsprop_conv2_up_0_kernel_rms_read_readvariableop6savev2_rmsprop_conv2_up_0_bias_rms_read_readvariableop6savev2_rmsprop_conv2_up_kernel_rms_read_readvariableop4savev2_rmsprop_conv2_up_bias_rms_read_readvariableop1savev2_rmsprop_up2_kernel_rms_read_readvariableop/savev2_rmsprop_up2_bias_rms_read_readvariableop8savev2_rmsprop_conv1_up_0_kernel_rms_read_readvariableop6savev2_rmsprop_conv1_up_0_bias_rms_read_readvariableop6savev2_rmsprop_conv1_up_kernel_rms_read_readvariableop4savev2_rmsprop_conv1_up_bias_rms_read_readvariableop5savev2_rmsprop_conv1_1_kernel_rms_read_readvariableop3savev2_rmsprop_conv1_1_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *b
dtypesX
V2T	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : :  : : @:@:@@:@:@?:?:??:?:??:?:??:?:??:?:??:?:??:?:?@:@:?@:@:@@:@:@ : :@ : :  : : :: : : : : : : : : : : : : :  : : @:@:@@:@:@?:?:??:?:??:?:??:?:??:?:??:?:??:?:?@:@:?@:@:@@:@:@ : :@ : :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-	)
'
_output_shapes
:@?:!


_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:-)
'
_output_shapes
:?@: 

_output_shapes
:@:-)
'
_output_shapes
:?@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@ : 

_output_shapes
: :,(
&
_output_shapes
:@ :  

_output_shapes
: :,!(
&
_output_shapes
:  : "

_output_shapes
: :,#(
&
_output_shapes
: : $

_output_shapes
::%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :,0(
&
_output_shapes
: : 1

_output_shapes
: :,2(
&
_output_shapes
:  : 3

_output_shapes
: :,4(
&
_output_shapes
: @: 5

_output_shapes
:@:,6(
&
_output_shapes
:@@: 7

_output_shapes
:@:-8)
'
_output_shapes
:@?:!9

_output_shapes	
:?:.:*
(
_output_shapes
:??:!;

_output_shapes	
:?:.<*
(
_output_shapes
:??:!=

_output_shapes	
:?:.>*
(
_output_shapes
:??:!?

_output_shapes	
:?:.@*
(
_output_shapes
:??:!A

_output_shapes	
:?:.B*
(
_output_shapes
:??:!C

_output_shapes	
:?:.D*
(
_output_shapes
:??:!E

_output_shapes	
:?:-F)
'
_output_shapes
:?@: G

_output_shapes
:@:-H)
'
_output_shapes
:?@: I

_output_shapes
:@:,J(
&
_output_shapes
:@@: K

_output_shapes
:@:,L(
&
_output_shapes
:@ : M

_output_shapes
: :,N(
&
_output_shapes
:@ : O

_output_shapes
: :,P(
&
_output_shapes
:  : Q

_output_shapes
: :,R(
&
_output_shapes
: : S

_output_shapes
::T

_output_shapes
: 
?
]
A__inference_pool3_layer_call_and_return_conditional_losses_203021

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_conv1_layer_call_and_return_conditional_losses_203161

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????PF 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????PF 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????PF : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????PF 
 
_user_specified_nameinputs
?
?
D__inference_conv2_up_layer_call_and_return_conditional_losses_203404

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(#@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????(#@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(#@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????(#@
 
_user_specified_nameinputs
?
U
)__inference_concat_1_layer_call_fn_205332
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_concat_1_layer_call_and_return_conditional_losses_2034352
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????PF@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:?????????PF :+??????????????????????????? :Y U
/
_output_shapes
:?????????PF 
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+??????????????????????????? 
"
_user_specified_name
inputs/1
?
?
)__inference_conv1_up_layer_call_fn_205368

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv1_up_layer_call_and_return_conditional_losses_2034652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????PF 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????PF : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????PF 
 
_user_specified_nameinputs
?
]
A__inference_pool2_layer_call_and_return_conditional_losses_203009

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
͎
?
A__inference_model_layer_call_and_return_conditional_losses_203489

inputs(
conv1_0_203145: 
conv1_0_203147: &
conv1_203162:  
conv1_203164: (
conv2_0_203180: @
conv2_0_203182:@&
conv2_203197:@@
conv2_203199:@)
conv3_0_203215:@?
conv3_0_203217:	?(
conv3_203232:??
conv3_203234:	?*
conv4_0_203257:??
conv4_0_203259:	?(
conv4_203274:??
conv4_203276:	?&

up4_203299:??

up4_203301:	?-
conv3_up_0_203326:?? 
conv3_up_0_203328:	?+
conv3_up_203343:??
conv3_up_203345:	?%

up3_203361:?@

up3_203363:@,
conv2_up_0_203388:?@
conv2_up_0_203390:@)
conv2_up_203405:@@
conv2_up_203407:@$

up2_203423:@ 

up2_203425: +
conv1_up_0_203449:@ 
conv1_up_0_203451: )
conv1_up_203466:  
conv1_up_203468: (
conv1_1_203482: 
conv1_1_203484:
identity??conv1/StatefulPartitionedCall?conv1_0/StatefulPartitionedCall?conv1_1/StatefulPartitionedCall? conv1_up/StatefulPartitionedCall?"conv1_up_0/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv2_0/StatefulPartitionedCall? conv2_up/StatefulPartitionedCall?"conv2_up_0/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv3_0/StatefulPartitionedCall? conv3_up/StatefulPartitionedCall?"conv3_up_0/StatefulPartitionedCall?conv4/StatefulPartitionedCall?conv4_0/StatefulPartitionedCall?up2/StatefulPartitionedCall?up3/StatefulPartitionedCall?up4/StatefulPartitionedCall?
zero_padding2d/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_zero_padding2d_layer_call_and_return_conditional_losses_2029852 
zero_padding2d/PartitionedCall?
conv1_0/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0conv1_0_203145conv1_0_203147*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv1_0_layer_call_and_return_conditional_losses_2031442!
conv1_0/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCall(conv1_0/StatefulPartitionedCall:output:0conv1_203162conv1_203164*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_2031612
conv1/StatefulPartitionedCall?
pool1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(# * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_pool1_layer_call_and_return_conditional_losses_2029972
pool1/PartitionedCall?
conv2_0/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0conv2_0_203180conv2_0_203182*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(#@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2_0_layer_call_and_return_conditional_losses_2031792!
conv2_0/StatefulPartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall(conv2_0/StatefulPartitionedCall:output:0conv2_203197conv2_203199*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(#@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_2031962
conv2/StatefulPartitionedCall?
pool2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_pool2_layer_call_and_return_conditional_losses_2030092
pool2/PartitionedCall?
conv3_0/StatefulPartitionedCallStatefulPartitionedCallpool2/PartitionedCall:output:0conv3_0_203215conv3_0_203217*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv3_0_layer_call_and_return_conditional_losses_2032142!
conv3_0/StatefulPartitionedCall?
conv3/StatefulPartitionedCallStatefulPartitionedCall(conv3_0/StatefulPartitionedCall:output:0conv3_203232conv3_203234*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_2032312
conv3/StatefulPartitionedCall?
drop3/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_drop3_layer_call_and_return_conditional_losses_2032422
drop3/PartitionedCall?
pool3/PartitionedCallPartitionedCalldrop3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_pool3_layer_call_and_return_conditional_losses_2030212
pool3/PartitionedCall?
conv4_0/StatefulPartitionedCallStatefulPartitionedCallpool3/PartitionedCall:output:0conv4_0_203257conv4_0_203259*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv4_0_layer_call_and_return_conditional_losses_2032562!
conv4_0/StatefulPartitionedCall?
conv4/StatefulPartitionedCallStatefulPartitionedCall(conv4_0/StatefulPartitionedCall:output:0conv4_203274conv4_203276*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv4_layer_call_and_return_conditional_losses_2032732
conv4/StatefulPartitionedCall?
drop4/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_drop4_layer_call_and_return_conditional_losses_2032842
drop4/PartitionedCall?
up4_0/PartitionedCallPartitionedCalldrop4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_up4_0_layer_call_and_return_conditional_losses_2030402
up4_0/PartitionedCall?
up4/StatefulPartitionedCallStatefulPartitionedCallup4_0/PartitionedCall:output:0
up4_203299
up4_203301*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_up4_layer_call_and_return_conditional_losses_2032982
up4/StatefulPartitionedCall?
 zero_padding2d_1/PartitionedCallPartitionedCall$up4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_2030532"
 zero_padding2d_1/PartitionedCall?
concat_3/PartitionedCallPartitionedCalldrop3/PartitionedCall:output:0)zero_padding2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_concat_3_layer_call_and_return_conditional_losses_2033122
concat_3/PartitionedCall?
"conv3_up_0/StatefulPartitionedCallStatefulPartitionedCall!concat_3/PartitionedCall:output:0conv3_up_0_203326conv3_up_0_203328*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv3_up_0_layer_call_and_return_conditional_losses_2033252$
"conv3_up_0/StatefulPartitionedCall?
 conv3_up/StatefulPartitionedCallStatefulPartitionedCall+conv3_up_0/StatefulPartitionedCall:output:0conv3_up_203343conv3_up_203345*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv3_up_layer_call_and_return_conditional_losses_2033422"
 conv3_up/StatefulPartitionedCall?
up3_0/PartitionedCallPartitionedCall)conv3_up/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_up3_0_layer_call_and_return_conditional_losses_2030722
up3_0/PartitionedCall?
up3/StatefulPartitionedCallStatefulPartitionedCallup3_0/PartitionedCall:output:0
up3_203361
up3_203363*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_up3_layer_call_and_return_conditional_losses_2033602
up3/StatefulPartitionedCall?
 zero_padding2d_2/PartitionedCallPartitionedCall$up3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_zero_padding2d_2_layer_call_and_return_conditional_losses_2030852"
 zero_padding2d_2/PartitionedCall?
concat_2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0)zero_padding2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????(#?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_concat_2_layer_call_and_return_conditional_losses_2033742
concat_2/PartitionedCall?
"conv2_up_0/StatefulPartitionedCallStatefulPartitionedCall!concat_2/PartitionedCall:output:0conv2_up_0_203388conv2_up_0_203390*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(#@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2_up_0_layer_call_and_return_conditional_losses_2033872$
"conv2_up_0/StatefulPartitionedCall?
 conv2_up/StatefulPartitionedCallStatefulPartitionedCall+conv2_up_0/StatefulPartitionedCall:output:0conv2_up_203405conv2_up_203407*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(#@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2_up_layer_call_and_return_conditional_losses_2034042"
 conv2_up/StatefulPartitionedCall?
up2_0/PartitionedCallPartitionedCall)conv2_up/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_up2_0_layer_call_and_return_conditional_losses_2031042
up2_0/PartitionedCall?
up2/StatefulPartitionedCallStatefulPartitionedCallup2_0/PartitionedCall:output:0
up2_203423
up2_203425*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_up2_layer_call_and_return_conditional_losses_2034222
up2/StatefulPartitionedCall?
concat_1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0$up2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_concat_1_layer_call_and_return_conditional_losses_2034352
concat_1/PartitionedCall?
"conv1_up_0/StatefulPartitionedCallStatefulPartitionedCall!concat_1/PartitionedCall:output:0conv1_up_0_203449conv1_up_0_203451*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv1_up_0_layer_call_and_return_conditional_losses_2034482$
"conv1_up_0/StatefulPartitionedCall?
 conv1_up/StatefulPartitionedCallStatefulPartitionedCall+conv1_up_0/StatefulPartitionedCall:output:0conv1_up_203466conv1_up_203468*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv1_up_layer_call_and_return_conditional_losses_2034652"
 conv1_up/StatefulPartitionedCall?
conv1_1/StatefulPartitionedCallStatefulPartitionedCall)conv1_up/StatefulPartitionedCall:output:0conv1_1_203482conv1_1_203484*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv1_1_layer_call_and_return_conditional_losses_2034812!
conv1_1/StatefulPartitionedCall?
cropping2d/PartitionedCallPartitionedCall(conv1_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????OE* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_cropping2d_layer_call_and_return_conditional_losses_2031192
cropping2d/PartitionedCall?
IdentityIdentity#cropping2d/PartitionedCall:output:0^conv1/StatefulPartitionedCall ^conv1_0/StatefulPartitionedCall ^conv1_1/StatefulPartitionedCall!^conv1_up/StatefulPartitionedCall#^conv1_up_0/StatefulPartitionedCall^conv2/StatefulPartitionedCall ^conv2_0/StatefulPartitionedCall!^conv2_up/StatefulPartitionedCall#^conv2_up_0/StatefulPartitionedCall^conv3/StatefulPartitionedCall ^conv3_0/StatefulPartitionedCall!^conv3_up/StatefulPartitionedCall#^conv3_up_0/StatefulPartitionedCall^conv4/StatefulPartitionedCall ^conv4_0/StatefulPartitionedCall^up2/StatefulPartitionedCall^up3/StatefulPartitionedCall^up4/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????OE2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????OE: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2B
conv1_0/StatefulPartitionedCallconv1_0/StatefulPartitionedCall2B
conv1_1/StatefulPartitionedCallconv1_1/StatefulPartitionedCall2D
 conv1_up/StatefulPartitionedCall conv1_up/StatefulPartitionedCall2H
"conv1_up_0/StatefulPartitionedCall"conv1_up_0/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2B
conv2_0/StatefulPartitionedCallconv2_0/StatefulPartitionedCall2D
 conv2_up/StatefulPartitionedCall conv2_up/StatefulPartitionedCall2H
"conv2_up_0/StatefulPartitionedCall"conv2_up_0/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2B
conv3_0/StatefulPartitionedCallconv3_0/StatefulPartitionedCall2D
 conv3_up/StatefulPartitionedCall conv3_up/StatefulPartitionedCall2H
"conv3_up_0/StatefulPartitionedCall"conv3_up_0/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2B
conv4_0/StatefulPartitionedCallconv4_0/StatefulPartitionedCall2:
up2/StatefulPartitionedCallup2/StatefulPartitionedCall2:
up3/StatefulPartitionedCallup3/StatefulPartitionedCall2:
up4/StatefulPartitionedCallup4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????OE
 
_user_specified_nameinputs
?
?
A__inference_conv4_layer_call_and_return_conditional_losses_205133

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????
?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????
?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????
?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
?
 
_user_specified_nameinputs
??
?
A__inference_model_layer_call_and_return_conditional_losses_204369
input_1(
conv1_0_204264: 
conv1_0_204266: &
conv1_204269:  
conv1_204271: (
conv2_0_204275: @
conv2_0_204277:@&
conv2_204280:@@
conv2_204282:@)
conv3_0_204286:@?
conv3_0_204288:	?(
conv3_204291:??
conv3_204293:	?*
conv4_0_204298:??
conv4_0_204300:	?(
conv4_204303:??
conv4_204305:	?&

up4_204310:??

up4_204312:	?-
conv3_up_0_204317:?? 
conv3_up_0_204319:	?+
conv3_up_204322:??
conv3_up_204324:	?%

up3_204328:?@

up3_204330:@,
conv2_up_0_204335:?@
conv2_up_0_204337:@)
conv2_up_204340:@@
conv2_up_204342:@$

up2_204346:@ 

up2_204348: +
conv1_up_0_204352:@ 
conv1_up_0_204354: )
conv1_up_204357:  
conv1_up_204359: (
conv1_1_204362: 
conv1_1_204364:
identity??conv1/StatefulPartitionedCall?conv1_0/StatefulPartitionedCall?conv1_1/StatefulPartitionedCall? conv1_up/StatefulPartitionedCall?"conv1_up_0/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv2_0/StatefulPartitionedCall? conv2_up/StatefulPartitionedCall?"conv2_up_0/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv3_0/StatefulPartitionedCall? conv3_up/StatefulPartitionedCall?"conv3_up_0/StatefulPartitionedCall?conv4/StatefulPartitionedCall?conv4_0/StatefulPartitionedCall?drop3/StatefulPartitionedCall?drop4/StatefulPartitionedCall?up2/StatefulPartitionedCall?up3/StatefulPartitionedCall?up4/StatefulPartitionedCall?
zero_padding2d/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_zero_padding2d_layer_call_and_return_conditional_losses_2029852 
zero_padding2d/PartitionedCall?
conv1_0/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0conv1_0_204264conv1_0_204266*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv1_0_layer_call_and_return_conditional_losses_2031442!
conv1_0/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCall(conv1_0/StatefulPartitionedCall:output:0conv1_204269conv1_204271*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_2031612
conv1/StatefulPartitionedCall?
pool1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(# * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_pool1_layer_call_and_return_conditional_losses_2029972
pool1/PartitionedCall?
conv2_0/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0conv2_0_204275conv2_0_204277*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(#@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2_0_layer_call_and_return_conditional_losses_2031792!
conv2_0/StatefulPartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall(conv2_0/StatefulPartitionedCall:output:0conv2_204280conv2_204282*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(#@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_2031962
conv2/StatefulPartitionedCall?
pool2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_pool2_layer_call_and_return_conditional_losses_2030092
pool2/PartitionedCall?
conv3_0/StatefulPartitionedCallStatefulPartitionedCallpool2/PartitionedCall:output:0conv3_0_204286conv3_0_204288*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv3_0_layer_call_and_return_conditional_losses_2032142!
conv3_0/StatefulPartitionedCall?
conv3/StatefulPartitionedCallStatefulPartitionedCall(conv3_0/StatefulPartitionedCall:output:0conv3_204291conv3_204293*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_2032312
conv3/StatefulPartitionedCall?
drop3/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_drop3_layer_call_and_return_conditional_losses_2037482
drop3/StatefulPartitionedCall?
pool3/PartitionedCallPartitionedCall&drop3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_pool3_layer_call_and_return_conditional_losses_2030212
pool3/PartitionedCall?
conv4_0/StatefulPartitionedCallStatefulPartitionedCallpool3/PartitionedCall:output:0conv4_0_204298conv4_0_204300*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv4_0_layer_call_and_return_conditional_losses_2032562!
conv4_0/StatefulPartitionedCall?
conv4/StatefulPartitionedCallStatefulPartitionedCall(conv4_0/StatefulPartitionedCall:output:0conv4_204303conv4_204305*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv4_layer_call_and_return_conditional_losses_2032732
conv4/StatefulPartitionedCall?
drop4/StatefulPartitionedCallStatefulPartitionedCall&conv4/StatefulPartitionedCall:output:0^drop3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_drop4_layer_call_and_return_conditional_losses_2037052
drop4/StatefulPartitionedCall?
up4_0/PartitionedCallPartitionedCall&drop4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_up4_0_layer_call_and_return_conditional_losses_2030402
up4_0/PartitionedCall?
up4/StatefulPartitionedCallStatefulPartitionedCallup4_0/PartitionedCall:output:0
up4_204310
up4_204312*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_up4_layer_call_and_return_conditional_losses_2032982
up4/StatefulPartitionedCall?
 zero_padding2d_1/PartitionedCallPartitionedCall$up4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_2030532"
 zero_padding2d_1/PartitionedCall?
concat_3/PartitionedCallPartitionedCall&drop3/StatefulPartitionedCall:output:0)zero_padding2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_concat_3_layer_call_and_return_conditional_losses_2033122
concat_3/PartitionedCall?
"conv3_up_0/StatefulPartitionedCallStatefulPartitionedCall!concat_3/PartitionedCall:output:0conv3_up_0_204317conv3_up_0_204319*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv3_up_0_layer_call_and_return_conditional_losses_2033252$
"conv3_up_0/StatefulPartitionedCall?
 conv3_up/StatefulPartitionedCallStatefulPartitionedCall+conv3_up_0/StatefulPartitionedCall:output:0conv3_up_204322conv3_up_204324*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv3_up_layer_call_and_return_conditional_losses_2033422"
 conv3_up/StatefulPartitionedCall?
up3_0/PartitionedCallPartitionedCall)conv3_up/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_up3_0_layer_call_and_return_conditional_losses_2030722
up3_0/PartitionedCall?
up3/StatefulPartitionedCallStatefulPartitionedCallup3_0/PartitionedCall:output:0
up3_204328
up3_204330*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_up3_layer_call_and_return_conditional_losses_2033602
up3/StatefulPartitionedCall?
 zero_padding2d_2/PartitionedCallPartitionedCall$up3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_zero_padding2d_2_layer_call_and_return_conditional_losses_2030852"
 zero_padding2d_2/PartitionedCall?
concat_2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0)zero_padding2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????(#?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_concat_2_layer_call_and_return_conditional_losses_2033742
concat_2/PartitionedCall?
"conv2_up_0/StatefulPartitionedCallStatefulPartitionedCall!concat_2/PartitionedCall:output:0conv2_up_0_204335conv2_up_0_204337*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(#@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2_up_0_layer_call_and_return_conditional_losses_2033872$
"conv2_up_0/StatefulPartitionedCall?
 conv2_up/StatefulPartitionedCallStatefulPartitionedCall+conv2_up_0/StatefulPartitionedCall:output:0conv2_up_204340conv2_up_204342*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(#@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2_up_layer_call_and_return_conditional_losses_2034042"
 conv2_up/StatefulPartitionedCall?
up2_0/PartitionedCallPartitionedCall)conv2_up/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_up2_0_layer_call_and_return_conditional_losses_2031042
up2_0/PartitionedCall?
up2/StatefulPartitionedCallStatefulPartitionedCallup2_0/PartitionedCall:output:0
up2_204346
up2_204348*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_up2_layer_call_and_return_conditional_losses_2034222
up2/StatefulPartitionedCall?
concat_1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0$up2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_concat_1_layer_call_and_return_conditional_losses_2034352
concat_1/PartitionedCall?
"conv1_up_0/StatefulPartitionedCallStatefulPartitionedCall!concat_1/PartitionedCall:output:0conv1_up_0_204352conv1_up_0_204354*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv1_up_0_layer_call_and_return_conditional_losses_2034482$
"conv1_up_0/StatefulPartitionedCall?
 conv1_up/StatefulPartitionedCallStatefulPartitionedCall+conv1_up_0/StatefulPartitionedCall:output:0conv1_up_204357conv1_up_204359*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv1_up_layer_call_and_return_conditional_losses_2034652"
 conv1_up/StatefulPartitionedCall?
conv1_1/StatefulPartitionedCallStatefulPartitionedCall)conv1_up/StatefulPartitionedCall:output:0conv1_1_204362conv1_1_204364*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv1_1_layer_call_and_return_conditional_losses_2034812!
conv1_1/StatefulPartitionedCall?
cropping2d/PartitionedCallPartitionedCall(conv1_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????OE* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_cropping2d_layer_call_and_return_conditional_losses_2031192
cropping2d/PartitionedCall?
IdentityIdentity#cropping2d/PartitionedCall:output:0^conv1/StatefulPartitionedCall ^conv1_0/StatefulPartitionedCall ^conv1_1/StatefulPartitionedCall!^conv1_up/StatefulPartitionedCall#^conv1_up_0/StatefulPartitionedCall^conv2/StatefulPartitionedCall ^conv2_0/StatefulPartitionedCall!^conv2_up/StatefulPartitionedCall#^conv2_up_0/StatefulPartitionedCall^conv3/StatefulPartitionedCall ^conv3_0/StatefulPartitionedCall!^conv3_up/StatefulPartitionedCall#^conv3_up_0/StatefulPartitionedCall^conv4/StatefulPartitionedCall ^conv4_0/StatefulPartitionedCall^drop3/StatefulPartitionedCall^drop4/StatefulPartitionedCall^up2/StatefulPartitionedCall^up3/StatefulPartitionedCall^up4/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????OE2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????OE: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2B
conv1_0/StatefulPartitionedCallconv1_0/StatefulPartitionedCall2B
conv1_1/StatefulPartitionedCallconv1_1/StatefulPartitionedCall2D
 conv1_up/StatefulPartitionedCall conv1_up/StatefulPartitionedCall2H
"conv1_up_0/StatefulPartitionedCall"conv1_up_0/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2B
conv2_0/StatefulPartitionedCallconv2_0/StatefulPartitionedCall2D
 conv2_up/StatefulPartitionedCall conv2_up/StatefulPartitionedCall2H
"conv2_up_0/StatefulPartitionedCall"conv2_up_0/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2B
conv3_0/StatefulPartitionedCallconv3_0/StatefulPartitionedCall2D
 conv3_up/StatefulPartitionedCall conv3_up/StatefulPartitionedCall2H
"conv3_up_0/StatefulPartitionedCall"conv3_up_0/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2B
conv4_0/StatefulPartitionedCallconv4_0/StatefulPartitionedCall2>
drop3/StatefulPartitionedCalldrop3/StatefulPartitionedCall2>
drop4/StatefulPartitionedCalldrop4/StatefulPartitionedCall2:
up2/StatefulPartitionedCallup2/StatefulPartitionedCall2:
up3/StatefulPartitionedCallup3/StatefulPartitionedCall2:
up4/StatefulPartitionedCallup4/StatefulPartitionedCall:X T
/
_output_shapes
:?????????OE
!
_user_specified_name	input_1
?
`
A__inference_drop4_layer_call_and_return_conditional_losses_205160

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????
?2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????
?*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????
?2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????
?2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????
?2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????
?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????
?:X T
0
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?
?
C__inference_conv1_0_layer_call_and_return_conditional_losses_203144

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????PF 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????PF 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????PF: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????PF
 
_user_specified_nameinputs
?
?
(__inference_conv4_0_layer_call_fn_205102

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv4_0_layer_call_and_return_conditional_losses_2032562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????
?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????
?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?
?
+__inference_conv1_up_0_layer_call_fn_205348

inputs!
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv1_up_0_layer_call_and_return_conditional_losses_2034482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????PF 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????PF@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????PF@
 
_user_specified_nameinputs
?
]
A__inference_up4_0_layer_call_and_return_conditional_losses_203040

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?
A__inference_model_layer_call_and_return_conditional_losses_203999

inputs(
conv1_0_203894: 
conv1_0_203896: &
conv1_203899:  
conv1_203901: (
conv2_0_203905: @
conv2_0_203907:@&
conv2_203910:@@
conv2_203912:@)
conv3_0_203916:@?
conv3_0_203918:	?(
conv3_203921:??
conv3_203923:	?*
conv4_0_203928:??
conv4_0_203930:	?(
conv4_203933:??
conv4_203935:	?&

up4_203940:??

up4_203942:	?-
conv3_up_0_203947:?? 
conv3_up_0_203949:	?+
conv3_up_203952:??
conv3_up_203954:	?%

up3_203958:?@

up3_203960:@,
conv2_up_0_203965:?@
conv2_up_0_203967:@)
conv2_up_203970:@@
conv2_up_203972:@$

up2_203976:@ 

up2_203978: +
conv1_up_0_203982:@ 
conv1_up_0_203984: )
conv1_up_203987:  
conv1_up_203989: (
conv1_1_203992: 
conv1_1_203994:
identity??conv1/StatefulPartitionedCall?conv1_0/StatefulPartitionedCall?conv1_1/StatefulPartitionedCall? conv1_up/StatefulPartitionedCall?"conv1_up_0/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv2_0/StatefulPartitionedCall? conv2_up/StatefulPartitionedCall?"conv2_up_0/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv3_0/StatefulPartitionedCall? conv3_up/StatefulPartitionedCall?"conv3_up_0/StatefulPartitionedCall?conv4/StatefulPartitionedCall?conv4_0/StatefulPartitionedCall?drop3/StatefulPartitionedCall?drop4/StatefulPartitionedCall?up2/StatefulPartitionedCall?up3/StatefulPartitionedCall?up4/StatefulPartitionedCall?
zero_padding2d/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_zero_padding2d_layer_call_and_return_conditional_losses_2029852 
zero_padding2d/PartitionedCall?
conv1_0/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0conv1_0_203894conv1_0_203896*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv1_0_layer_call_and_return_conditional_losses_2031442!
conv1_0/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCall(conv1_0/StatefulPartitionedCall:output:0conv1_203899conv1_203901*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_2031612
conv1/StatefulPartitionedCall?
pool1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(# * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_pool1_layer_call_and_return_conditional_losses_2029972
pool1/PartitionedCall?
conv2_0/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0conv2_0_203905conv2_0_203907*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(#@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2_0_layer_call_and_return_conditional_losses_2031792!
conv2_0/StatefulPartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall(conv2_0/StatefulPartitionedCall:output:0conv2_203910conv2_203912*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(#@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_2031962
conv2/StatefulPartitionedCall?
pool2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_pool2_layer_call_and_return_conditional_losses_2030092
pool2/PartitionedCall?
conv3_0/StatefulPartitionedCallStatefulPartitionedCallpool2/PartitionedCall:output:0conv3_0_203916conv3_0_203918*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv3_0_layer_call_and_return_conditional_losses_2032142!
conv3_0/StatefulPartitionedCall?
conv3/StatefulPartitionedCallStatefulPartitionedCall(conv3_0/StatefulPartitionedCall:output:0conv3_203921conv3_203923*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_2032312
conv3/StatefulPartitionedCall?
drop3/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_drop3_layer_call_and_return_conditional_losses_2037482
drop3/StatefulPartitionedCall?
pool3/PartitionedCallPartitionedCall&drop3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_pool3_layer_call_and_return_conditional_losses_2030212
pool3/PartitionedCall?
conv4_0/StatefulPartitionedCallStatefulPartitionedCallpool3/PartitionedCall:output:0conv4_0_203928conv4_0_203930*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv4_0_layer_call_and_return_conditional_losses_2032562!
conv4_0/StatefulPartitionedCall?
conv4/StatefulPartitionedCallStatefulPartitionedCall(conv4_0/StatefulPartitionedCall:output:0conv4_203933conv4_203935*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv4_layer_call_and_return_conditional_losses_2032732
conv4/StatefulPartitionedCall?
drop4/StatefulPartitionedCallStatefulPartitionedCall&conv4/StatefulPartitionedCall:output:0^drop3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_drop4_layer_call_and_return_conditional_losses_2037052
drop4/StatefulPartitionedCall?
up4_0/PartitionedCallPartitionedCall&drop4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_up4_0_layer_call_and_return_conditional_losses_2030402
up4_0/PartitionedCall?
up4/StatefulPartitionedCallStatefulPartitionedCallup4_0/PartitionedCall:output:0
up4_203940
up4_203942*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_up4_layer_call_and_return_conditional_losses_2032982
up4/StatefulPartitionedCall?
 zero_padding2d_1/PartitionedCallPartitionedCall$up4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_2030532"
 zero_padding2d_1/PartitionedCall?
concat_3/PartitionedCallPartitionedCall&drop3/StatefulPartitionedCall:output:0)zero_padding2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_concat_3_layer_call_and_return_conditional_losses_2033122
concat_3/PartitionedCall?
"conv3_up_0/StatefulPartitionedCallStatefulPartitionedCall!concat_3/PartitionedCall:output:0conv3_up_0_203947conv3_up_0_203949*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv3_up_0_layer_call_and_return_conditional_losses_2033252$
"conv3_up_0/StatefulPartitionedCall?
 conv3_up/StatefulPartitionedCallStatefulPartitionedCall+conv3_up_0/StatefulPartitionedCall:output:0conv3_up_203952conv3_up_203954*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv3_up_layer_call_and_return_conditional_losses_2033422"
 conv3_up/StatefulPartitionedCall?
up3_0/PartitionedCallPartitionedCall)conv3_up/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_up3_0_layer_call_and_return_conditional_losses_2030722
up3_0/PartitionedCall?
up3/StatefulPartitionedCallStatefulPartitionedCallup3_0/PartitionedCall:output:0
up3_203958
up3_203960*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_up3_layer_call_and_return_conditional_losses_2033602
up3/StatefulPartitionedCall?
 zero_padding2d_2/PartitionedCallPartitionedCall$up3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_zero_padding2d_2_layer_call_and_return_conditional_losses_2030852"
 zero_padding2d_2/PartitionedCall?
concat_2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0)zero_padding2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????(#?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_concat_2_layer_call_and_return_conditional_losses_2033742
concat_2/PartitionedCall?
"conv2_up_0/StatefulPartitionedCallStatefulPartitionedCall!concat_2/PartitionedCall:output:0conv2_up_0_203965conv2_up_0_203967*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(#@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2_up_0_layer_call_and_return_conditional_losses_2033872$
"conv2_up_0/StatefulPartitionedCall?
 conv2_up/StatefulPartitionedCallStatefulPartitionedCall+conv2_up_0/StatefulPartitionedCall:output:0conv2_up_203970conv2_up_203972*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(#@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2_up_layer_call_and_return_conditional_losses_2034042"
 conv2_up/StatefulPartitionedCall?
up2_0/PartitionedCallPartitionedCall)conv2_up/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_up2_0_layer_call_and_return_conditional_losses_2031042
up2_0/PartitionedCall?
up2/StatefulPartitionedCallStatefulPartitionedCallup2_0/PartitionedCall:output:0
up2_203976
up2_203978*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_up2_layer_call_and_return_conditional_losses_2034222
up2/StatefulPartitionedCall?
concat_1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0$up2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_concat_1_layer_call_and_return_conditional_losses_2034352
concat_1/PartitionedCall?
"conv1_up_0/StatefulPartitionedCallStatefulPartitionedCall!concat_1/PartitionedCall:output:0conv1_up_0_203982conv1_up_0_203984*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv1_up_0_layer_call_and_return_conditional_losses_2034482$
"conv1_up_0/StatefulPartitionedCall?
 conv1_up/StatefulPartitionedCallStatefulPartitionedCall+conv1_up_0/StatefulPartitionedCall:output:0conv1_up_203987conv1_up_203989*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv1_up_layer_call_and_return_conditional_losses_2034652"
 conv1_up/StatefulPartitionedCall?
conv1_1/StatefulPartitionedCallStatefulPartitionedCall)conv1_up/StatefulPartitionedCall:output:0conv1_1_203992conv1_1_203994*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv1_1_layer_call_and_return_conditional_losses_2034812!
conv1_1/StatefulPartitionedCall?
cropping2d/PartitionedCallPartitionedCall(conv1_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????OE* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_cropping2d_layer_call_and_return_conditional_losses_2031192
cropping2d/PartitionedCall?
IdentityIdentity#cropping2d/PartitionedCall:output:0^conv1/StatefulPartitionedCall ^conv1_0/StatefulPartitionedCall ^conv1_1/StatefulPartitionedCall!^conv1_up/StatefulPartitionedCall#^conv1_up_0/StatefulPartitionedCall^conv2/StatefulPartitionedCall ^conv2_0/StatefulPartitionedCall!^conv2_up/StatefulPartitionedCall#^conv2_up_0/StatefulPartitionedCall^conv3/StatefulPartitionedCall ^conv3_0/StatefulPartitionedCall!^conv3_up/StatefulPartitionedCall#^conv3_up_0/StatefulPartitionedCall^conv4/StatefulPartitionedCall ^conv4_0/StatefulPartitionedCall^drop3/StatefulPartitionedCall^drop4/StatefulPartitionedCall^up2/StatefulPartitionedCall^up3/StatefulPartitionedCall^up4/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????OE2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????OE: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2B
conv1_0/StatefulPartitionedCallconv1_0/StatefulPartitionedCall2B
conv1_1/StatefulPartitionedCallconv1_1/StatefulPartitionedCall2D
 conv1_up/StatefulPartitionedCall conv1_up/StatefulPartitionedCall2H
"conv1_up_0/StatefulPartitionedCall"conv1_up_0/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2B
conv2_0/StatefulPartitionedCallconv2_0/StatefulPartitionedCall2D
 conv2_up/StatefulPartitionedCall conv2_up/StatefulPartitionedCall2H
"conv2_up_0/StatefulPartitionedCall"conv2_up_0/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2B
conv3_0/StatefulPartitionedCallconv3_0/StatefulPartitionedCall2D
 conv3_up/StatefulPartitionedCall conv3_up/StatefulPartitionedCall2H
"conv3_up_0/StatefulPartitionedCall"conv3_up_0/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2B
conv4_0/StatefulPartitionedCallconv4_0/StatefulPartitionedCall2>
drop3/StatefulPartitionedCalldrop3/StatefulPartitionedCall2>
drop4/StatefulPartitionedCalldrop4/StatefulPartitionedCall2:
up2/StatefulPartitionedCallup2/StatefulPartitionedCall2:
up3/StatefulPartitionedCallup3/StatefulPartitionedCall2:
up4/StatefulPartitionedCallup4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????OE
 
_user_specified_nameinputs
?
_
&__inference_drop4_layer_call_fn_205143

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_drop4_layer_call_and_return_conditional_losses_2037052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????
?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????
?22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?
U
)__inference_concat_3_layer_call_fn_205186
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_concat_3_layer_call_and_return_conditional_losses_2033122
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:??????????:,????????????????????????????:Z V
0
_output_shapes
:??????????
"
_user_specified_name
inputs/0:lh
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/1
?
M
1__inference_zero_padding2d_2_layer_call_fn_203091

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_zero_padding2d_2_layer_call_and_return_conditional_losses_2030852
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
_
&__inference_drop3_layer_call_fn_205076

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_drop3_layer_call_and_return_conditional_losses_2037482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?	
&__inference_model_layer_call_fn_204531

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?%

unknown_21:?@

unknown_22:@%

unknown_23:?@

unknown_24:@$

unknown_25:@@

unknown_26:@$

unknown_27:@ 

unknown_28: $

unknown_29:@ 

unknown_30: $

unknown_31:  

unknown_32: $

unknown_33: 

unknown_34:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????OE*F
_read_only_resource_inputs(
&$	
 !"#$*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2034892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????OE2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????OE: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????OE
 
_user_specified_nameinputs
?
?
$__inference_up2_layer_call_fn_205315

inputs!
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_up2_layer_call_and_return_conditional_losses_2034222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
?__inference_up3_layer_call_and_return_conditional_losses_203360

inputs9
conv2d_readvariableop_resource:?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_conv1_1_layer_call_fn_205388

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv1_1_layer_call_and_return_conditional_losses_2034812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????PF2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????PF : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????PF 
 
_user_specified_nameinputs
?
?
A__inference_conv2_layer_call_and_return_conditional_losses_205026

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(#@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????(#@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(#@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????(#@
 
_user_specified_nameinputs
?
p
D__inference_concat_1_layer_call_and_return_conditional_losses_205339
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????PF@2
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:?????????PF@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:?????????PF :+??????????????????????????? :Y U
/
_output_shapes
:?????????PF 
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+??????????????????????????? 
"
_user_specified_name
inputs/1
?
`
A__inference_drop3_layer_call_and_return_conditional_losses_205093

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
]
A__inference_up3_0_layer_call_and_return_conditional_losses_203072

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
]
A__inference_pool1_layer_call_and_return_conditional_losses_202997

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
n
D__inference_concat_2_layer_call_and_return_conditional_losses_203374

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????(#?2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????(#?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:?????????(#@:+???????????????????????????@:W S
/
_output_shapes
:?????????(#@
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
K
/__inference_zero_padding2d_layer_call_fn_202991

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_zero_padding2d_layer_call_and_return_conditional_losses_2029852
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_conv4_0_layer_call_and_return_conditional_losses_205113

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????
?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????
?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????
?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?

?
C__inference_conv1_1_layer_call_and_return_conditional_losses_203481

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????PF2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????PF : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????PF 
 
_user_specified_nameinputs
?
?
$__inference_up4_layer_call_fn_205169

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_up4_layer_call_and_return_conditional_losses_2032982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_conv4_0_layer_call_and_return_conditional_losses_203256

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????
?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????
?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????
?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?
?
?__inference_up2_layer_call_and_return_conditional_losses_203422

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
(__inference_conv1_0_layer_call_fn_204955

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv1_0_layer_call_and_return_conditional_losses_2031442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????PF 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????PF: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????PF
 
_user_specified_nameinputs
?
]
A__inference_up2_0_layer_call_and_return_conditional_losses_203104

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_conv2_layer_call_and_return_conditional_losses_203196

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(#@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????(#@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(#@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????(#@
 
_user_specified_nameinputs
?
`
A__inference_drop4_layer_call_and_return_conditional_losses_203705

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????
?2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????
?*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????
?2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????
?2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????
?2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????
?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????
?:X T
0
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?
?
C__inference_conv2_0_layer_call_and_return_conditional_losses_205006

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(#@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????(#@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(# : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????(# 
 
_user_specified_nameinputs
?
?	
&__inference_model_layer_call_fn_204151
input_1!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?%

unknown_21:?@

unknown_22:@%

unknown_23:?@

unknown_24:@$

unknown_25:@@

unknown_26:@$

unknown_27:@ 

unknown_28: $

unknown_29:@ 

unknown_30: $

unknown_31:  

unknown_32: $

unknown_33: 

unknown_34:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????OE*F
_read_only_resource_inputs(
&$	
 !"#$*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2039992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????OE2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????OE: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????OE
!
_user_specified_name	input_1
?
?
C__inference_conv3_0_layer_call_and_return_conditional_losses_203214

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
D__inference_conv3_up_layer_call_and_return_conditional_losses_205233

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_conv3_up_0_layer_call_fn_205202

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv3_up_0_layer_call_and_return_conditional_losses_2033252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
A__inference_drop3_layer_call_and_return_conditional_losses_203242

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
B
&__inference_pool3_layer_call_fn_203027

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_pool3_layer_call_and_return_conditional_losses_2030212
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
?__inference_up4_layer_call_and_return_conditional_losses_205180

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?	
&__inference_model_layer_call_fn_204608

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?%

unknown_21:?@

unknown_22:@%

unknown_23:?@

unknown_24:@$

unknown_25:@@

unknown_26:@$

unknown_27:@ 

unknown_28: $

unknown_29:@ 

unknown_30: $

unknown_31:  

unknown_32: $

unknown_33: 

unknown_34:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????OE*F
_read_only_resource_inputs(
&$	
 !"#$*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2039992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????OE2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????OE: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????OE
 
_user_specified_nameinputs
?
?
C__inference_conv1_0_layer_call_and_return_conditional_losses_204966

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????PF 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????PF 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????PF: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????PF
 
_user_specified_nameinputs
??
?
A__inference_model_layer_call_and_return_conditional_losses_204770

inputs@
&conv1_0_conv2d_readvariableop_resource: 5
'conv1_0_biasadd_readvariableop_resource: >
$conv1_conv2d_readvariableop_resource:  3
%conv1_biasadd_readvariableop_resource: @
&conv2_0_conv2d_readvariableop_resource: @5
'conv2_0_biasadd_readvariableop_resource:@>
$conv2_conv2d_readvariableop_resource:@@3
%conv2_biasadd_readvariableop_resource:@A
&conv3_0_conv2d_readvariableop_resource:@?6
'conv3_0_biasadd_readvariableop_resource:	?@
$conv3_conv2d_readvariableop_resource:??4
%conv3_biasadd_readvariableop_resource:	?B
&conv4_0_conv2d_readvariableop_resource:??6
'conv4_0_biasadd_readvariableop_resource:	?@
$conv4_conv2d_readvariableop_resource:??4
%conv4_biasadd_readvariableop_resource:	?>
"up4_conv2d_readvariableop_resource:??2
#up4_biasadd_readvariableop_resource:	?E
)conv3_up_0_conv2d_readvariableop_resource:??9
*conv3_up_0_biasadd_readvariableop_resource:	?C
'conv3_up_conv2d_readvariableop_resource:??7
(conv3_up_biasadd_readvariableop_resource:	?=
"up3_conv2d_readvariableop_resource:?@1
#up3_biasadd_readvariableop_resource:@D
)conv2_up_0_conv2d_readvariableop_resource:?@8
*conv2_up_0_biasadd_readvariableop_resource:@A
'conv2_up_conv2d_readvariableop_resource:@@6
(conv2_up_biasadd_readvariableop_resource:@<
"up2_conv2d_readvariableop_resource:@ 1
#up2_biasadd_readvariableop_resource: C
)conv1_up_0_conv2d_readvariableop_resource:@ 8
*conv1_up_0_biasadd_readvariableop_resource: A
'conv1_up_conv2d_readvariableop_resource:  6
(conv1_up_biasadd_readvariableop_resource: @
&conv1_1_conv2d_readvariableop_resource: 5
'conv1_1_biasadd_readvariableop_resource:
identity??conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv1_0/BiasAdd/ReadVariableOp?conv1_0/Conv2D/ReadVariableOp?conv1_1/BiasAdd/ReadVariableOp?conv1_1/Conv2D/ReadVariableOp?conv1_up/BiasAdd/ReadVariableOp?conv1_up/Conv2D/ReadVariableOp?!conv1_up_0/BiasAdd/ReadVariableOp? conv1_up_0/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp?conv2_0/BiasAdd/ReadVariableOp?conv2_0/Conv2D/ReadVariableOp?conv2_up/BiasAdd/ReadVariableOp?conv2_up/Conv2D/ReadVariableOp?!conv2_up_0/BiasAdd/ReadVariableOp? conv2_up_0/Conv2D/ReadVariableOp?conv3/BiasAdd/ReadVariableOp?conv3/Conv2D/ReadVariableOp?conv3_0/BiasAdd/ReadVariableOp?conv3_0/Conv2D/ReadVariableOp?conv3_up/BiasAdd/ReadVariableOp?conv3_up/Conv2D/ReadVariableOp?!conv3_up_0/BiasAdd/ReadVariableOp? conv3_up_0/Conv2D/ReadVariableOp?conv4/BiasAdd/ReadVariableOp?conv4/Conv2D/ReadVariableOp?conv4_0/BiasAdd/ReadVariableOp?conv4_0/Conv2D/ReadVariableOp?up2/BiasAdd/ReadVariableOp?up2/Conv2D/ReadVariableOp?up3/BiasAdd/ReadVariableOp?up3/Conv2D/ReadVariableOp?up4/BiasAdd/ReadVariableOp?up4/Conv2D/ReadVariableOp?
zero_padding2d/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               2
zero_padding2d/Pad/paddings?
zero_padding2d/PadPadinputs$zero_padding2d/Pad/paddings:output:0*
T0*/
_output_shapes
:?????????PF2
zero_padding2d/Pad?
conv1_0/Conv2D/ReadVariableOpReadVariableOp&conv1_0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv1_0/Conv2D/ReadVariableOp?
conv1_0/Conv2DConv2Dzero_padding2d/Pad:output:0%conv1_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF *
paddingSAME*
strides
2
conv1_0/Conv2D?
conv1_0/BiasAdd/ReadVariableOpReadVariableOp'conv1_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
conv1_0/BiasAdd/ReadVariableOp?
conv1_0/BiasAddBiasAddconv1_0/Conv2D:output:0&conv1_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF 2
conv1_0/BiasAddx
conv1_0/ReluReluconv1_0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????PF 2
conv1_0/Relu?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv1/Conv2D/ReadVariableOp?
conv1/Conv2DConv2Dconv1_0/Relu:activations:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF *
paddingSAME*
strides
2
conv1/Conv2D?
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv1/BiasAdd/ReadVariableOp?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF 2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????PF 2

conv1/Relu?
pool1/MaxPoolMaxPoolconv1/Relu:activations:0*/
_output_shapes
:?????????(# *
ksize
*
paddingVALID*
strides
2
pool1/MaxPool?
conv2_0/Conv2D/ReadVariableOpReadVariableOp&conv2_0_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv2_0/Conv2D/ReadVariableOp?
conv2_0/Conv2DConv2Dpool1/MaxPool:output:0%conv2_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@*
paddingSAME*
strides
2
conv2_0/Conv2D?
conv2_0/BiasAdd/ReadVariableOpReadVariableOp'conv2_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
conv2_0/BiasAdd/ReadVariableOp?
conv2_0/BiasAddBiasAddconv2_0/Conv2D:output:0&conv2_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@2
conv2_0/BiasAddx
conv2_0/ReluReluconv2_0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(#@2
conv2_0/Relu?
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
conv2/Conv2D/ReadVariableOp?
conv2/Conv2DConv2Dconv2_0/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@*
paddingSAME*
strides
2
conv2/Conv2D?
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2/BiasAdd/ReadVariableOp?
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(#@2

conv2/Relu?
pool2/MaxPoolMaxPoolconv2/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
pool2/MaxPool?
conv3_0/Conv2D/ReadVariableOpReadVariableOp&conv3_0_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
conv3_0/Conv2D/ReadVariableOp?
conv3_0/Conv2DConv2Dpool2/MaxPool:output:0%conv3_0/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv3_0/Conv2D?
conv3_0/BiasAdd/ReadVariableOpReadVariableOp'conv3_0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv3_0/BiasAdd/ReadVariableOp?
conv3_0/BiasAddBiasAddconv3_0/Conv2D:output:0&conv3_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv3_0/BiasAddy
conv3_0/ReluReluconv3_0/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv3_0/Relu?
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv3/Conv2D/ReadVariableOp?
conv3/Conv2DConv2Dconv3_0/Relu:activations:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv3/Conv2D?
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv3/BiasAdd/ReadVariableOp?
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv3/BiasAdds

conv3/ReluReluconv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2

conv3/Relu?
drop3/IdentityIdentityconv3/Relu:activations:0*
T0*0
_output_shapes
:??????????2
drop3/Identity?
pool3/MaxPoolMaxPooldrop3/Identity:output:0*0
_output_shapes
:?????????
?*
ksize
*
paddingVALID*
strides
2
pool3/MaxPool?
conv4_0/Conv2D/ReadVariableOpReadVariableOp&conv4_0_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv4_0/Conv2D/ReadVariableOp?
conv4_0/Conv2DConv2Dpool3/MaxPool:output:0%conv4_0/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?*
paddingSAME*
strides
2
conv4_0/Conv2D?
conv4_0/BiasAdd/ReadVariableOpReadVariableOp'conv4_0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv4_0/BiasAdd/ReadVariableOp?
conv4_0/BiasAddBiasAddconv4_0/Conv2D:output:0&conv4_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?2
conv4_0/BiasAddy
conv4_0/ReluReluconv4_0/BiasAdd:output:0*
T0*0
_output_shapes
:?????????
?2
conv4_0/Relu?
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv4/Conv2D/ReadVariableOp?
conv4/Conv2DConv2Dconv4_0/Relu:activations:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?*
paddingSAME*
strides
2
conv4/Conv2D?
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv4/BiasAdd/ReadVariableOp?
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?2
conv4/BiasAdds

conv4/ReluReluconv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????
?2

conv4/Relu?
drop4/IdentityIdentityconv4/Relu:activations:0*
T0*0
_output_shapes
:?????????
?2
drop4/Identityk
up4_0/ConstConst*
_output_shapes
:*
dtype0*
valueB"
      2
up4_0/Consto
up4_0/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up4_0/Const_1p
	up4_0/mulMulup4_0/Const:output:0up4_0/Const_1:output:0*
T0*
_output_shapes
:2
	up4_0/mul?
"up4_0/resize/ResizeNearestNeighborResizeNearestNeighbordrop4/Identity:output:0up4_0/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2$
"up4_0/resize/ResizeNearestNeighbor?
up4/Conv2D/ReadVariableOpReadVariableOp"up4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
up4/Conv2D/ReadVariableOp?

up4/Conv2DConv2D3up4_0/resize/ResizeNearestNeighbor:resized_images:0!up4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2

up4/Conv2D?
up4/BiasAdd/ReadVariableOpReadVariableOp#up4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
up4/BiasAdd/ReadVariableOp?
up4/BiasAddBiasAddup4/Conv2D:output:0"up4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
up4/BiasAddm
up4/ReluReluup4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2

up4/Relu?
zero_padding2d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                2
zero_padding2d_1/Pad/paddings?
zero_padding2d_1/PadPadup4/Relu:activations:0&zero_padding2d_1/Pad/paddings:output:0*
T0*0
_output_shapes
:??????????2
zero_padding2d_1/Padn
concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_3/concat/axis?
concat_3/concatConcatV2drop3/Identity:output:0zero_padding2d_1/Pad:output:0concat_3/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
concat_3/concat?
 conv3_up_0/Conv2D/ReadVariableOpReadVariableOp)conv3_up_0_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02"
 conv3_up_0/Conv2D/ReadVariableOp?
conv3_up_0/Conv2DConv2Dconcat_3/concat:output:0(conv3_up_0/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv3_up_0/Conv2D?
!conv3_up_0/BiasAdd/ReadVariableOpReadVariableOp*conv3_up_0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!conv3_up_0/BiasAdd/ReadVariableOp?
conv3_up_0/BiasAddBiasAddconv3_up_0/Conv2D:output:0)conv3_up_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv3_up_0/BiasAdd?
conv3_up_0/ReluReluconv3_up_0/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv3_up_0/Relu?
conv3_up/Conv2D/ReadVariableOpReadVariableOp'conv3_up_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv3_up/Conv2D/ReadVariableOp?
conv3_up/Conv2DConv2Dconv3_up_0/Relu:activations:0&conv3_up/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv3_up/Conv2D?
conv3_up/BiasAdd/ReadVariableOpReadVariableOp(conv3_up_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv3_up/BiasAdd/ReadVariableOp?
conv3_up/BiasAddBiasAddconv3_up/Conv2D:output:0'conv3_up/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv3_up/BiasAdd|
conv3_up/ReluReluconv3_up/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv3_up/Reluk
up3_0/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up3_0/Consto
up3_0/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up3_0/Const_1p
	up3_0/mulMulup3_0/Const:output:0up3_0/Const_1:output:0*
T0*
_output_shapes
:2
	up3_0/mul?
"up3_0/resize/ResizeNearestNeighborResizeNearestNeighborconv3_up/Relu:activations:0up3_0/mul:z:0*
T0*0
_output_shapes
:?????????("?*
half_pixel_centers(2$
"up3_0/resize/ResizeNearestNeighbor?
up3/Conv2D/ReadVariableOpReadVariableOp"up3_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
up3/Conv2D/ReadVariableOp?

up3/Conv2DConv2D3up3_0/resize/ResizeNearestNeighbor:resized_images:0!up3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????("@*
paddingSAME*
strides
2

up3/Conv2D?
up3/BiasAdd/ReadVariableOpReadVariableOp#up3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
up3/BiasAdd/ReadVariableOp?
up3/BiasAddBiasAddup3/Conv2D:output:0"up3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????("@2
up3/BiasAddl
up3/ReluReluup3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????("@2

up3/Relu?
zero_padding2d_2/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                2
zero_padding2d_2/Pad/paddings?
zero_padding2d_2/PadPadup3/Relu:activations:0&zero_padding2d_2/Pad/paddings:output:0*
T0*/
_output_shapes
:?????????(#@2
zero_padding2d_2/Padn
concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/concat/axis?
concat_2/concatConcatV2conv2/Relu:activations:0zero_padding2d_2/Pad:output:0concat_2/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????(#?2
concat_2/concat?
 conv2_up_0/Conv2D/ReadVariableOpReadVariableOp)conv2_up_0_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02"
 conv2_up_0/Conv2D/ReadVariableOp?
conv2_up_0/Conv2DConv2Dconcat_2/concat:output:0(conv2_up_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@*
paddingSAME*
strides
2
conv2_up_0/Conv2D?
!conv2_up_0/BiasAdd/ReadVariableOpReadVariableOp*conv2_up_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!conv2_up_0/BiasAdd/ReadVariableOp?
conv2_up_0/BiasAddBiasAddconv2_up_0/Conv2D:output:0)conv2_up_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@2
conv2_up_0/BiasAdd?
conv2_up_0/ReluReluconv2_up_0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(#@2
conv2_up_0/Relu?
conv2_up/Conv2D/ReadVariableOpReadVariableOp'conv2_up_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2_up/Conv2D/ReadVariableOp?
conv2_up/Conv2DConv2Dconv2_up_0/Relu:activations:0&conv2_up/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@*
paddingSAME*
strides
2
conv2_up/Conv2D?
conv2_up/BiasAdd/ReadVariableOpReadVariableOp(conv2_up_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2_up/BiasAdd/ReadVariableOp?
conv2_up/BiasAddBiasAddconv2_up/Conv2D:output:0'conv2_up/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@2
conv2_up/BiasAdd{
conv2_up/ReluReluconv2_up/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(#@2
conv2_up/Reluk
up2_0/ConstConst*
_output_shapes
:*
dtype0*
valueB"(   #   2
up2_0/Consto
up2_0/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up2_0/Const_1p
	up2_0/mulMulup2_0/Const:output:0up2_0/Const_1:output:0*
T0*
_output_shapes
:2
	up2_0/mul?
"up2_0/resize/ResizeNearestNeighborResizeNearestNeighborconv2_up/Relu:activations:0up2_0/mul:z:0*
T0*/
_output_shapes
:?????????PF@*
half_pixel_centers(2$
"up2_0/resize/ResizeNearestNeighbor?
up2/Conv2D/ReadVariableOpReadVariableOp"up2_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
up2/Conv2D/ReadVariableOp?

up2/Conv2DConv2D3up2_0/resize/ResizeNearestNeighbor:resized_images:0!up2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF *
paddingSAME*
strides
2

up2/Conv2D?
up2/BiasAdd/ReadVariableOpReadVariableOp#up2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
up2/BiasAdd/ReadVariableOp?
up2/BiasAddBiasAddup2/Conv2D:output:0"up2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF 2
up2/BiasAddl
up2/ReluReluup2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????PF 2

up2/Relun
concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/concat/axis?
concat_1/concatConcatV2conv1/Relu:activations:0up2/Relu:activations:0concat_1/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????PF@2
concat_1/concat?
 conv1_up_0/Conv2D/ReadVariableOpReadVariableOp)conv1_up_0_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02"
 conv1_up_0/Conv2D/ReadVariableOp?
conv1_up_0/Conv2DConv2Dconcat_1/concat:output:0(conv1_up_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF *
paddingSAME*
strides
2
conv1_up_0/Conv2D?
!conv1_up_0/BiasAdd/ReadVariableOpReadVariableOp*conv1_up_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!conv1_up_0/BiasAdd/ReadVariableOp?
conv1_up_0/BiasAddBiasAddconv1_up_0/Conv2D:output:0)conv1_up_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF 2
conv1_up_0/BiasAdd?
conv1_up_0/ReluReluconv1_up_0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????PF 2
conv1_up_0/Relu?
conv1_up/Conv2D/ReadVariableOpReadVariableOp'conv1_up_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv1_up/Conv2D/ReadVariableOp?
conv1_up/Conv2DConv2Dconv1_up_0/Relu:activations:0&conv1_up/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF *
paddingSAME*
strides
2
conv1_up/Conv2D?
conv1_up/BiasAdd/ReadVariableOpReadVariableOp(conv1_up_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1_up/BiasAdd/ReadVariableOp?
conv1_up/BiasAddBiasAddconv1_up/Conv2D:output:0'conv1_up/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF 2
conv1_up/BiasAdd{
conv1_up/ReluReluconv1_up/BiasAdd:output:0*
T0*/
_output_shapes
:?????????PF 2
conv1_up/Relu?
conv1_1/Conv2D/ReadVariableOpReadVariableOp&conv1_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv1_1/Conv2D/ReadVariableOp?
conv1_1/Conv2DConv2Dconv1_up/Relu:activations:0%conv1_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF*
paddingVALID*
strides
2
conv1_1/Conv2D?
conv1_1/BiasAdd/ReadVariableOpReadVariableOp'conv1_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
conv1_1/BiasAdd/ReadVariableOp?
conv1_1/BiasAddBiasAddconv1_1/Conv2D:output:0&conv1_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF2
conv1_1/BiasAdd?
cropping2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2 
cropping2d/strided_slice/stack?
 cropping2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"    ????????    2"
 cropping2d/strided_slice/stack_1?
 cropping2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2"
 cropping2d/strided_slice/stack_2?
cropping2d/strided_sliceStridedSliceconv1_1/BiasAdd:output:0'cropping2d/strided_slice/stack:output:0)cropping2d/strided_slice/stack_1:output:0)cropping2d/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????OE*

begin_mask	*
end_mask	2
cropping2d/strided_slice?	
IdentityIdentity!cropping2d/strided_slice:output:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv1_0/BiasAdd/ReadVariableOp^conv1_0/Conv2D/ReadVariableOp^conv1_1/BiasAdd/ReadVariableOp^conv1_1/Conv2D/ReadVariableOp ^conv1_up/BiasAdd/ReadVariableOp^conv1_up/Conv2D/ReadVariableOp"^conv1_up_0/BiasAdd/ReadVariableOp!^conv1_up_0/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv2_0/BiasAdd/ReadVariableOp^conv2_0/Conv2D/ReadVariableOp ^conv2_up/BiasAdd/ReadVariableOp^conv2_up/Conv2D/ReadVariableOp"^conv2_up_0/BiasAdd/ReadVariableOp!^conv2_up_0/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp^conv3_0/BiasAdd/ReadVariableOp^conv3_0/Conv2D/ReadVariableOp ^conv3_up/BiasAdd/ReadVariableOp^conv3_up/Conv2D/ReadVariableOp"^conv3_up_0/BiasAdd/ReadVariableOp!^conv3_up_0/Conv2D/ReadVariableOp^conv4/BiasAdd/ReadVariableOp^conv4/Conv2D/ReadVariableOp^conv4_0/BiasAdd/ReadVariableOp^conv4_0/Conv2D/ReadVariableOp^up2/BiasAdd/ReadVariableOp^up2/Conv2D/ReadVariableOp^up3/BiasAdd/ReadVariableOp^up3/Conv2D/ReadVariableOp^up4/BiasAdd/ReadVariableOp^up4/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????OE2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????OE: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2@
conv1_0/BiasAdd/ReadVariableOpconv1_0/BiasAdd/ReadVariableOp2>
conv1_0/Conv2D/ReadVariableOpconv1_0/Conv2D/ReadVariableOp2@
conv1_1/BiasAdd/ReadVariableOpconv1_1/BiasAdd/ReadVariableOp2>
conv1_1/Conv2D/ReadVariableOpconv1_1/Conv2D/ReadVariableOp2B
conv1_up/BiasAdd/ReadVariableOpconv1_up/BiasAdd/ReadVariableOp2@
conv1_up/Conv2D/ReadVariableOpconv1_up/Conv2D/ReadVariableOp2F
!conv1_up_0/BiasAdd/ReadVariableOp!conv1_up_0/BiasAdd/ReadVariableOp2D
 conv1_up_0/Conv2D/ReadVariableOp conv1_up_0/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2@
conv2_0/BiasAdd/ReadVariableOpconv2_0/BiasAdd/ReadVariableOp2>
conv2_0/Conv2D/ReadVariableOpconv2_0/Conv2D/ReadVariableOp2B
conv2_up/BiasAdd/ReadVariableOpconv2_up/BiasAdd/ReadVariableOp2@
conv2_up/Conv2D/ReadVariableOpconv2_up/Conv2D/ReadVariableOp2F
!conv2_up_0/BiasAdd/ReadVariableOp!conv2_up_0/BiasAdd/ReadVariableOp2D
 conv2_up_0/Conv2D/ReadVariableOp conv2_up_0/Conv2D/ReadVariableOp2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2@
conv3_0/BiasAdd/ReadVariableOpconv3_0/BiasAdd/ReadVariableOp2>
conv3_0/Conv2D/ReadVariableOpconv3_0/Conv2D/ReadVariableOp2B
conv3_up/BiasAdd/ReadVariableOpconv3_up/BiasAdd/ReadVariableOp2@
conv3_up/Conv2D/ReadVariableOpconv3_up/Conv2D/ReadVariableOp2F
!conv3_up_0/BiasAdd/ReadVariableOp!conv3_up_0/BiasAdd/ReadVariableOp2D
 conv3_up_0/Conv2D/ReadVariableOp conv3_up_0/Conv2D/ReadVariableOp2<
conv4/BiasAdd/ReadVariableOpconv4/BiasAdd/ReadVariableOp2:
conv4/Conv2D/ReadVariableOpconv4/Conv2D/ReadVariableOp2@
conv4_0/BiasAdd/ReadVariableOpconv4_0/BiasAdd/ReadVariableOp2>
conv4_0/Conv2D/ReadVariableOpconv4_0/Conv2D/ReadVariableOp28
up2/BiasAdd/ReadVariableOpup2/BiasAdd/ReadVariableOp26
up2/Conv2D/ReadVariableOpup2/Conv2D/ReadVariableOp28
up3/BiasAdd/ReadVariableOpup3/BiasAdd/ReadVariableOp26
up3/Conv2D/ReadVariableOpup3/Conv2D/ReadVariableOp28
up4/BiasAdd/ReadVariableOpup4/BiasAdd/ReadVariableOp26
up4/Conv2D/ReadVariableOpup4/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????OE
 
_user_specified_nameinputs
?
?
C__inference_conv2_0_layer_call_and_return_conditional_losses_203179

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(#@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????(#@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(# : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????(# 
 
_user_specified_nameinputs
?
?
A__inference_conv3_layer_call_and_return_conditional_losses_205066

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
A__inference_model_layer_call_and_return_conditional_losses_204946

inputs@
&conv1_0_conv2d_readvariableop_resource: 5
'conv1_0_biasadd_readvariableop_resource: >
$conv1_conv2d_readvariableop_resource:  3
%conv1_biasadd_readvariableop_resource: @
&conv2_0_conv2d_readvariableop_resource: @5
'conv2_0_biasadd_readvariableop_resource:@>
$conv2_conv2d_readvariableop_resource:@@3
%conv2_biasadd_readvariableop_resource:@A
&conv3_0_conv2d_readvariableop_resource:@?6
'conv3_0_biasadd_readvariableop_resource:	?@
$conv3_conv2d_readvariableop_resource:??4
%conv3_biasadd_readvariableop_resource:	?B
&conv4_0_conv2d_readvariableop_resource:??6
'conv4_0_biasadd_readvariableop_resource:	?@
$conv4_conv2d_readvariableop_resource:??4
%conv4_biasadd_readvariableop_resource:	?>
"up4_conv2d_readvariableop_resource:??2
#up4_biasadd_readvariableop_resource:	?E
)conv3_up_0_conv2d_readvariableop_resource:??9
*conv3_up_0_biasadd_readvariableop_resource:	?C
'conv3_up_conv2d_readvariableop_resource:??7
(conv3_up_biasadd_readvariableop_resource:	?=
"up3_conv2d_readvariableop_resource:?@1
#up3_biasadd_readvariableop_resource:@D
)conv2_up_0_conv2d_readvariableop_resource:?@8
*conv2_up_0_biasadd_readvariableop_resource:@A
'conv2_up_conv2d_readvariableop_resource:@@6
(conv2_up_biasadd_readvariableop_resource:@<
"up2_conv2d_readvariableop_resource:@ 1
#up2_biasadd_readvariableop_resource: C
)conv1_up_0_conv2d_readvariableop_resource:@ 8
*conv1_up_0_biasadd_readvariableop_resource: A
'conv1_up_conv2d_readvariableop_resource:  6
(conv1_up_biasadd_readvariableop_resource: @
&conv1_1_conv2d_readvariableop_resource: 5
'conv1_1_biasadd_readvariableop_resource:
identity??conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv1_0/BiasAdd/ReadVariableOp?conv1_0/Conv2D/ReadVariableOp?conv1_1/BiasAdd/ReadVariableOp?conv1_1/Conv2D/ReadVariableOp?conv1_up/BiasAdd/ReadVariableOp?conv1_up/Conv2D/ReadVariableOp?!conv1_up_0/BiasAdd/ReadVariableOp? conv1_up_0/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp?conv2_0/BiasAdd/ReadVariableOp?conv2_0/Conv2D/ReadVariableOp?conv2_up/BiasAdd/ReadVariableOp?conv2_up/Conv2D/ReadVariableOp?!conv2_up_0/BiasAdd/ReadVariableOp? conv2_up_0/Conv2D/ReadVariableOp?conv3/BiasAdd/ReadVariableOp?conv3/Conv2D/ReadVariableOp?conv3_0/BiasAdd/ReadVariableOp?conv3_0/Conv2D/ReadVariableOp?conv3_up/BiasAdd/ReadVariableOp?conv3_up/Conv2D/ReadVariableOp?!conv3_up_0/BiasAdd/ReadVariableOp? conv3_up_0/Conv2D/ReadVariableOp?conv4/BiasAdd/ReadVariableOp?conv4/Conv2D/ReadVariableOp?conv4_0/BiasAdd/ReadVariableOp?conv4_0/Conv2D/ReadVariableOp?up2/BiasAdd/ReadVariableOp?up2/Conv2D/ReadVariableOp?up3/BiasAdd/ReadVariableOp?up3/Conv2D/ReadVariableOp?up4/BiasAdd/ReadVariableOp?up4/Conv2D/ReadVariableOp?
zero_padding2d/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               2
zero_padding2d/Pad/paddings?
zero_padding2d/PadPadinputs$zero_padding2d/Pad/paddings:output:0*
T0*/
_output_shapes
:?????????PF2
zero_padding2d/Pad?
conv1_0/Conv2D/ReadVariableOpReadVariableOp&conv1_0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv1_0/Conv2D/ReadVariableOp?
conv1_0/Conv2DConv2Dzero_padding2d/Pad:output:0%conv1_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF *
paddingSAME*
strides
2
conv1_0/Conv2D?
conv1_0/BiasAdd/ReadVariableOpReadVariableOp'conv1_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
conv1_0/BiasAdd/ReadVariableOp?
conv1_0/BiasAddBiasAddconv1_0/Conv2D:output:0&conv1_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF 2
conv1_0/BiasAddx
conv1_0/ReluReluconv1_0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????PF 2
conv1_0/Relu?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv1/Conv2D/ReadVariableOp?
conv1/Conv2DConv2Dconv1_0/Relu:activations:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF *
paddingSAME*
strides
2
conv1/Conv2D?
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv1/BiasAdd/ReadVariableOp?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF 2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????PF 2

conv1/Relu?
pool1/MaxPoolMaxPoolconv1/Relu:activations:0*/
_output_shapes
:?????????(# *
ksize
*
paddingVALID*
strides
2
pool1/MaxPool?
conv2_0/Conv2D/ReadVariableOpReadVariableOp&conv2_0_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv2_0/Conv2D/ReadVariableOp?
conv2_0/Conv2DConv2Dpool1/MaxPool:output:0%conv2_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@*
paddingSAME*
strides
2
conv2_0/Conv2D?
conv2_0/BiasAdd/ReadVariableOpReadVariableOp'conv2_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
conv2_0/BiasAdd/ReadVariableOp?
conv2_0/BiasAddBiasAddconv2_0/Conv2D:output:0&conv2_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@2
conv2_0/BiasAddx
conv2_0/ReluReluconv2_0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(#@2
conv2_0/Relu?
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
conv2/Conv2D/ReadVariableOp?
conv2/Conv2DConv2Dconv2_0/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@*
paddingSAME*
strides
2
conv2/Conv2D?
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2/BiasAdd/ReadVariableOp?
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(#@2

conv2/Relu?
pool2/MaxPoolMaxPoolconv2/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
pool2/MaxPool?
conv3_0/Conv2D/ReadVariableOpReadVariableOp&conv3_0_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
conv3_0/Conv2D/ReadVariableOp?
conv3_0/Conv2DConv2Dpool2/MaxPool:output:0%conv3_0/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv3_0/Conv2D?
conv3_0/BiasAdd/ReadVariableOpReadVariableOp'conv3_0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv3_0/BiasAdd/ReadVariableOp?
conv3_0/BiasAddBiasAddconv3_0/Conv2D:output:0&conv3_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv3_0/BiasAddy
conv3_0/ReluReluconv3_0/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv3_0/Relu?
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv3/Conv2D/ReadVariableOp?
conv3/Conv2DConv2Dconv3_0/Relu:activations:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv3/Conv2D?
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv3/BiasAdd/ReadVariableOp?
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv3/BiasAdds

conv3/ReluReluconv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2

conv3/Reluo
drop3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
drop3/dropout/Const?
drop3/dropout/MulMulconv3/Relu:activations:0drop3/dropout/Const:output:0*
T0*0
_output_shapes
:??????????2
drop3/dropout/Mulr
drop3/dropout/ShapeShapeconv3/Relu:activations:0*
T0*
_output_shapes
:2
drop3/dropout/Shape?
*drop3/dropout/random_uniform/RandomUniformRandomUniformdrop3/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02,
*drop3/dropout/random_uniform/RandomUniform?
drop3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
drop3/dropout/GreaterEqual/y?
drop3/dropout/GreaterEqualGreaterEqual3drop3/dropout/random_uniform/RandomUniform:output:0%drop3/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2
drop3/dropout/GreaterEqual?
drop3/dropout/CastCastdrop3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
drop3/dropout/Cast?
drop3/dropout/Mul_1Muldrop3/dropout/Mul:z:0drop3/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
drop3/dropout/Mul_1?
pool3/MaxPoolMaxPooldrop3/dropout/Mul_1:z:0*0
_output_shapes
:?????????
?*
ksize
*
paddingVALID*
strides
2
pool3/MaxPool?
conv4_0/Conv2D/ReadVariableOpReadVariableOp&conv4_0_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv4_0/Conv2D/ReadVariableOp?
conv4_0/Conv2DConv2Dpool3/MaxPool:output:0%conv4_0/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?*
paddingSAME*
strides
2
conv4_0/Conv2D?
conv4_0/BiasAdd/ReadVariableOpReadVariableOp'conv4_0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
conv4_0/BiasAdd/ReadVariableOp?
conv4_0/BiasAddBiasAddconv4_0/Conv2D:output:0&conv4_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?2
conv4_0/BiasAddy
conv4_0/ReluReluconv4_0/BiasAdd:output:0*
T0*0
_output_shapes
:?????????
?2
conv4_0/Relu?
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv4/Conv2D/ReadVariableOp?
conv4/Conv2DConv2Dconv4_0/Relu:activations:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?*
paddingSAME*
strides
2
conv4/Conv2D?
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv4/BiasAdd/ReadVariableOp?
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?2
conv4/BiasAdds

conv4/ReluReluconv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????
?2

conv4/Reluo
drop4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
drop4/dropout/Const?
drop4/dropout/MulMulconv4/Relu:activations:0drop4/dropout/Const:output:0*
T0*0
_output_shapes
:?????????
?2
drop4/dropout/Mulr
drop4/dropout/ShapeShapeconv4/Relu:activations:0*
T0*
_output_shapes
:2
drop4/dropout/Shape?
*drop4/dropout/random_uniform/RandomUniformRandomUniformdrop4/dropout/Shape:output:0*
T0*0
_output_shapes
:?????????
?*
dtype02,
*drop4/dropout/random_uniform/RandomUniform?
drop4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
drop4/dropout/GreaterEqual/y?
drop4/dropout/GreaterEqualGreaterEqual3drop4/dropout/random_uniform/RandomUniform:output:0%drop4/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????
?2
drop4/dropout/GreaterEqual?
drop4/dropout/CastCastdrop4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????
?2
drop4/dropout/Cast?
drop4/dropout/Mul_1Muldrop4/dropout/Mul:z:0drop4/dropout/Cast:y:0*
T0*0
_output_shapes
:?????????
?2
drop4/dropout/Mul_1k
up4_0/ConstConst*
_output_shapes
:*
dtype0*
valueB"
      2
up4_0/Consto
up4_0/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up4_0/Const_1p
	up4_0/mulMulup4_0/Const:output:0up4_0/Const_1:output:0*
T0*
_output_shapes
:2
	up4_0/mul?
"up4_0/resize/ResizeNearestNeighborResizeNearestNeighbordrop4/dropout/Mul_1:z:0up4_0/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2$
"up4_0/resize/ResizeNearestNeighbor?
up4/Conv2D/ReadVariableOpReadVariableOp"up4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
up4/Conv2D/ReadVariableOp?

up4/Conv2DConv2D3up4_0/resize/ResizeNearestNeighbor:resized_images:0!up4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2

up4/Conv2D?
up4/BiasAdd/ReadVariableOpReadVariableOp#up4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
up4/BiasAdd/ReadVariableOp?
up4/BiasAddBiasAddup4/Conv2D:output:0"up4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
up4/BiasAddm
up4/ReluReluup4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2

up4/Relu?
zero_padding2d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                2
zero_padding2d_1/Pad/paddings?
zero_padding2d_1/PadPadup4/Relu:activations:0&zero_padding2d_1/Pad/paddings:output:0*
T0*0
_output_shapes
:??????????2
zero_padding2d_1/Padn
concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_3/concat/axis?
concat_3/concatConcatV2drop3/dropout/Mul_1:z:0zero_padding2d_1/Pad:output:0concat_3/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
concat_3/concat?
 conv3_up_0/Conv2D/ReadVariableOpReadVariableOp)conv3_up_0_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02"
 conv3_up_0/Conv2D/ReadVariableOp?
conv3_up_0/Conv2DConv2Dconcat_3/concat:output:0(conv3_up_0/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv3_up_0/Conv2D?
!conv3_up_0/BiasAdd/ReadVariableOpReadVariableOp*conv3_up_0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02#
!conv3_up_0/BiasAdd/ReadVariableOp?
conv3_up_0/BiasAddBiasAddconv3_up_0/Conv2D:output:0)conv3_up_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv3_up_0/BiasAdd?
conv3_up_0/ReluReluconv3_up_0/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv3_up_0/Relu?
conv3_up/Conv2D/ReadVariableOpReadVariableOp'conv3_up_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv3_up/Conv2D/ReadVariableOp?
conv3_up/Conv2DConv2Dconv3_up_0/Relu:activations:0&conv3_up/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv3_up/Conv2D?
conv3_up/BiasAdd/ReadVariableOpReadVariableOp(conv3_up_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv3_up/BiasAdd/ReadVariableOp?
conv3_up/BiasAddBiasAddconv3_up/Conv2D:output:0'conv3_up/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv3_up/BiasAdd|
conv3_up/ReluReluconv3_up/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv3_up/Reluk
up3_0/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up3_0/Consto
up3_0/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up3_0/Const_1p
	up3_0/mulMulup3_0/Const:output:0up3_0/Const_1:output:0*
T0*
_output_shapes
:2
	up3_0/mul?
"up3_0/resize/ResizeNearestNeighborResizeNearestNeighborconv3_up/Relu:activations:0up3_0/mul:z:0*
T0*0
_output_shapes
:?????????("?*
half_pixel_centers(2$
"up3_0/resize/ResizeNearestNeighbor?
up3/Conv2D/ReadVariableOpReadVariableOp"up3_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
up3/Conv2D/ReadVariableOp?

up3/Conv2DConv2D3up3_0/resize/ResizeNearestNeighbor:resized_images:0!up3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????("@*
paddingSAME*
strides
2

up3/Conv2D?
up3/BiasAdd/ReadVariableOpReadVariableOp#up3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
up3/BiasAdd/ReadVariableOp?
up3/BiasAddBiasAddup3/Conv2D:output:0"up3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????("@2
up3/BiasAddl
up3/ReluReluup3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????("@2

up3/Relu?
zero_padding2d_2/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                2
zero_padding2d_2/Pad/paddings?
zero_padding2d_2/PadPadup3/Relu:activations:0&zero_padding2d_2/Pad/paddings:output:0*
T0*/
_output_shapes
:?????????(#@2
zero_padding2d_2/Padn
concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/concat/axis?
concat_2/concatConcatV2conv2/Relu:activations:0zero_padding2d_2/Pad:output:0concat_2/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????(#?2
concat_2/concat?
 conv2_up_0/Conv2D/ReadVariableOpReadVariableOp)conv2_up_0_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02"
 conv2_up_0/Conv2D/ReadVariableOp?
conv2_up_0/Conv2DConv2Dconcat_2/concat:output:0(conv2_up_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@*
paddingSAME*
strides
2
conv2_up_0/Conv2D?
!conv2_up_0/BiasAdd/ReadVariableOpReadVariableOp*conv2_up_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!conv2_up_0/BiasAdd/ReadVariableOp?
conv2_up_0/BiasAddBiasAddconv2_up_0/Conv2D:output:0)conv2_up_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@2
conv2_up_0/BiasAdd?
conv2_up_0/ReluReluconv2_up_0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(#@2
conv2_up_0/Relu?
conv2_up/Conv2D/ReadVariableOpReadVariableOp'conv2_up_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2_up/Conv2D/ReadVariableOp?
conv2_up/Conv2DConv2Dconv2_up_0/Relu:activations:0&conv2_up/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@*
paddingSAME*
strides
2
conv2_up/Conv2D?
conv2_up/BiasAdd/ReadVariableOpReadVariableOp(conv2_up_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2_up/BiasAdd/ReadVariableOp?
conv2_up/BiasAddBiasAddconv2_up/Conv2D:output:0'conv2_up/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@2
conv2_up/BiasAdd{
conv2_up/ReluReluconv2_up/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(#@2
conv2_up/Reluk
up2_0/ConstConst*
_output_shapes
:*
dtype0*
valueB"(   #   2
up2_0/Consto
up2_0/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up2_0/Const_1p
	up2_0/mulMulup2_0/Const:output:0up2_0/Const_1:output:0*
T0*
_output_shapes
:2
	up2_0/mul?
"up2_0/resize/ResizeNearestNeighborResizeNearestNeighborconv2_up/Relu:activations:0up2_0/mul:z:0*
T0*/
_output_shapes
:?????????PF@*
half_pixel_centers(2$
"up2_0/resize/ResizeNearestNeighbor?
up2/Conv2D/ReadVariableOpReadVariableOp"up2_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
up2/Conv2D/ReadVariableOp?

up2/Conv2DConv2D3up2_0/resize/ResizeNearestNeighbor:resized_images:0!up2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF *
paddingSAME*
strides
2

up2/Conv2D?
up2/BiasAdd/ReadVariableOpReadVariableOp#up2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
up2/BiasAdd/ReadVariableOp?
up2/BiasAddBiasAddup2/Conv2D:output:0"up2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF 2
up2/BiasAddl
up2/ReluReluup2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????PF 2

up2/Relun
concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/concat/axis?
concat_1/concatConcatV2conv1/Relu:activations:0up2/Relu:activations:0concat_1/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????PF@2
concat_1/concat?
 conv1_up_0/Conv2D/ReadVariableOpReadVariableOp)conv1_up_0_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02"
 conv1_up_0/Conv2D/ReadVariableOp?
conv1_up_0/Conv2DConv2Dconcat_1/concat:output:0(conv1_up_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF *
paddingSAME*
strides
2
conv1_up_0/Conv2D?
!conv1_up_0/BiasAdd/ReadVariableOpReadVariableOp*conv1_up_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!conv1_up_0/BiasAdd/ReadVariableOp?
conv1_up_0/BiasAddBiasAddconv1_up_0/Conv2D:output:0)conv1_up_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF 2
conv1_up_0/BiasAdd?
conv1_up_0/ReluReluconv1_up_0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????PF 2
conv1_up_0/Relu?
conv1_up/Conv2D/ReadVariableOpReadVariableOp'conv1_up_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv1_up/Conv2D/ReadVariableOp?
conv1_up/Conv2DConv2Dconv1_up_0/Relu:activations:0&conv1_up/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF *
paddingSAME*
strides
2
conv1_up/Conv2D?
conv1_up/BiasAdd/ReadVariableOpReadVariableOp(conv1_up_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1_up/BiasAdd/ReadVariableOp?
conv1_up/BiasAddBiasAddconv1_up/Conv2D:output:0'conv1_up/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF 2
conv1_up/BiasAdd{
conv1_up/ReluReluconv1_up/BiasAdd:output:0*
T0*/
_output_shapes
:?????????PF 2
conv1_up/Relu?
conv1_1/Conv2D/ReadVariableOpReadVariableOp&conv1_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv1_1/Conv2D/ReadVariableOp?
conv1_1/Conv2DConv2Dconv1_up/Relu:activations:0%conv1_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF*
paddingVALID*
strides
2
conv1_1/Conv2D?
conv1_1/BiasAdd/ReadVariableOpReadVariableOp'conv1_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
conv1_1/BiasAdd/ReadVariableOp?
conv1_1/BiasAddBiasAddconv1_1/Conv2D:output:0&conv1_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF2
conv1_1/BiasAdd?
cropping2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2 
cropping2d/strided_slice/stack?
 cropping2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"    ????????    2"
 cropping2d/strided_slice/stack_1?
 cropping2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2"
 cropping2d/strided_slice/stack_2?
cropping2d/strided_sliceStridedSliceconv1_1/BiasAdd:output:0'cropping2d/strided_slice/stack:output:0)cropping2d/strided_slice/stack_1:output:0)cropping2d/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????OE*

begin_mask	*
end_mask	2
cropping2d/strided_slice?	
IdentityIdentity!cropping2d/strided_slice:output:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv1_0/BiasAdd/ReadVariableOp^conv1_0/Conv2D/ReadVariableOp^conv1_1/BiasAdd/ReadVariableOp^conv1_1/Conv2D/ReadVariableOp ^conv1_up/BiasAdd/ReadVariableOp^conv1_up/Conv2D/ReadVariableOp"^conv1_up_0/BiasAdd/ReadVariableOp!^conv1_up_0/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv2_0/BiasAdd/ReadVariableOp^conv2_0/Conv2D/ReadVariableOp ^conv2_up/BiasAdd/ReadVariableOp^conv2_up/Conv2D/ReadVariableOp"^conv2_up_0/BiasAdd/ReadVariableOp!^conv2_up_0/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp^conv3_0/BiasAdd/ReadVariableOp^conv3_0/Conv2D/ReadVariableOp ^conv3_up/BiasAdd/ReadVariableOp^conv3_up/Conv2D/ReadVariableOp"^conv3_up_0/BiasAdd/ReadVariableOp!^conv3_up_0/Conv2D/ReadVariableOp^conv4/BiasAdd/ReadVariableOp^conv4/Conv2D/ReadVariableOp^conv4_0/BiasAdd/ReadVariableOp^conv4_0/Conv2D/ReadVariableOp^up2/BiasAdd/ReadVariableOp^up2/Conv2D/ReadVariableOp^up3/BiasAdd/ReadVariableOp^up3/Conv2D/ReadVariableOp^up4/BiasAdd/ReadVariableOp^up4/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????OE2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????OE: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2@
conv1_0/BiasAdd/ReadVariableOpconv1_0/BiasAdd/ReadVariableOp2>
conv1_0/Conv2D/ReadVariableOpconv1_0/Conv2D/ReadVariableOp2@
conv1_1/BiasAdd/ReadVariableOpconv1_1/BiasAdd/ReadVariableOp2>
conv1_1/Conv2D/ReadVariableOpconv1_1/Conv2D/ReadVariableOp2B
conv1_up/BiasAdd/ReadVariableOpconv1_up/BiasAdd/ReadVariableOp2@
conv1_up/Conv2D/ReadVariableOpconv1_up/Conv2D/ReadVariableOp2F
!conv1_up_0/BiasAdd/ReadVariableOp!conv1_up_0/BiasAdd/ReadVariableOp2D
 conv1_up_0/Conv2D/ReadVariableOp conv1_up_0/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2@
conv2_0/BiasAdd/ReadVariableOpconv2_0/BiasAdd/ReadVariableOp2>
conv2_0/Conv2D/ReadVariableOpconv2_0/Conv2D/ReadVariableOp2B
conv2_up/BiasAdd/ReadVariableOpconv2_up/BiasAdd/ReadVariableOp2@
conv2_up/Conv2D/ReadVariableOpconv2_up/Conv2D/ReadVariableOp2F
!conv2_up_0/BiasAdd/ReadVariableOp!conv2_up_0/BiasAdd/ReadVariableOp2D
 conv2_up_0/Conv2D/ReadVariableOp conv2_up_0/Conv2D/ReadVariableOp2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2@
conv3_0/BiasAdd/ReadVariableOpconv3_0/BiasAdd/ReadVariableOp2>
conv3_0/Conv2D/ReadVariableOpconv3_0/Conv2D/ReadVariableOp2B
conv3_up/BiasAdd/ReadVariableOpconv3_up/BiasAdd/ReadVariableOp2@
conv3_up/Conv2D/ReadVariableOpconv3_up/Conv2D/ReadVariableOp2F
!conv3_up_0/BiasAdd/ReadVariableOp!conv3_up_0/BiasAdd/ReadVariableOp2D
 conv3_up_0/Conv2D/ReadVariableOp conv3_up_0/Conv2D/ReadVariableOp2<
conv4/BiasAdd/ReadVariableOpconv4/BiasAdd/ReadVariableOp2:
conv4/Conv2D/ReadVariableOpconv4/Conv2D/ReadVariableOp2@
conv4_0/BiasAdd/ReadVariableOpconv4_0/BiasAdd/ReadVariableOp2>
conv4_0/Conv2D/ReadVariableOpconv4_0/Conv2D/ReadVariableOp28
up2/BiasAdd/ReadVariableOpup2/BiasAdd/ReadVariableOp26
up2/Conv2D/ReadVariableOpup2/Conv2D/ReadVariableOp28
up3/BiasAdd/ReadVariableOpup3/BiasAdd/ReadVariableOp26
up3/Conv2D/ReadVariableOpup3/Conv2D/ReadVariableOp28
up4/BiasAdd/ReadVariableOpup4/BiasAdd/ReadVariableOp26
up4/Conv2D/ReadVariableOpup4/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????OE
 
_user_specified_nameinputs
?
?
F__inference_conv2_up_0_layer_call_and_return_conditional_losses_205286

inputs9
conv2d_readvariableop_resource:?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(#@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????(#@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????(#?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????(#?
 
_user_specified_nameinputs
Ў
?
A__inference_model_layer_call_and_return_conditional_losses_204260
input_1(
conv1_0_204155: 
conv1_0_204157: &
conv1_204160:  
conv1_204162: (
conv2_0_204166: @
conv2_0_204168:@&
conv2_204171:@@
conv2_204173:@)
conv3_0_204177:@?
conv3_0_204179:	?(
conv3_204182:??
conv3_204184:	?*
conv4_0_204189:??
conv4_0_204191:	?(
conv4_204194:??
conv4_204196:	?&

up4_204201:??

up4_204203:	?-
conv3_up_0_204208:?? 
conv3_up_0_204210:	?+
conv3_up_204213:??
conv3_up_204215:	?%

up3_204219:?@

up3_204221:@,
conv2_up_0_204226:?@
conv2_up_0_204228:@)
conv2_up_204231:@@
conv2_up_204233:@$

up2_204237:@ 

up2_204239: +
conv1_up_0_204243:@ 
conv1_up_0_204245: )
conv1_up_204248:  
conv1_up_204250: (
conv1_1_204253: 
conv1_1_204255:
identity??conv1/StatefulPartitionedCall?conv1_0/StatefulPartitionedCall?conv1_1/StatefulPartitionedCall? conv1_up/StatefulPartitionedCall?"conv1_up_0/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv2_0/StatefulPartitionedCall? conv2_up/StatefulPartitionedCall?"conv2_up_0/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv3_0/StatefulPartitionedCall? conv3_up/StatefulPartitionedCall?"conv3_up_0/StatefulPartitionedCall?conv4/StatefulPartitionedCall?conv4_0/StatefulPartitionedCall?up2/StatefulPartitionedCall?up3/StatefulPartitionedCall?up4/StatefulPartitionedCall?
zero_padding2d/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_zero_padding2d_layer_call_and_return_conditional_losses_2029852 
zero_padding2d/PartitionedCall?
conv1_0/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0conv1_0_204155conv1_0_204157*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv1_0_layer_call_and_return_conditional_losses_2031442!
conv1_0/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCall(conv1_0/StatefulPartitionedCall:output:0conv1_204160conv1_204162*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_2031612
conv1/StatefulPartitionedCall?
pool1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(# * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_pool1_layer_call_and_return_conditional_losses_2029972
pool1/PartitionedCall?
conv2_0/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0conv2_0_204166conv2_0_204168*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(#@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2_0_layer_call_and_return_conditional_losses_2031792!
conv2_0/StatefulPartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall(conv2_0/StatefulPartitionedCall:output:0conv2_204171conv2_204173*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(#@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_2031962
conv2/StatefulPartitionedCall?
pool2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_pool2_layer_call_and_return_conditional_losses_2030092
pool2/PartitionedCall?
conv3_0/StatefulPartitionedCallStatefulPartitionedCallpool2/PartitionedCall:output:0conv3_0_204177conv3_0_204179*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv3_0_layer_call_and_return_conditional_losses_2032142!
conv3_0/StatefulPartitionedCall?
conv3/StatefulPartitionedCallStatefulPartitionedCall(conv3_0/StatefulPartitionedCall:output:0conv3_204182conv3_204184*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_2032312
conv3/StatefulPartitionedCall?
drop3/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_drop3_layer_call_and_return_conditional_losses_2032422
drop3/PartitionedCall?
pool3/PartitionedCallPartitionedCalldrop3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_pool3_layer_call_and_return_conditional_losses_2030212
pool3/PartitionedCall?
conv4_0/StatefulPartitionedCallStatefulPartitionedCallpool3/PartitionedCall:output:0conv4_0_204189conv4_0_204191*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv4_0_layer_call_and_return_conditional_losses_2032562!
conv4_0/StatefulPartitionedCall?
conv4/StatefulPartitionedCallStatefulPartitionedCall(conv4_0/StatefulPartitionedCall:output:0conv4_204194conv4_204196*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv4_layer_call_and_return_conditional_losses_2032732
conv4/StatefulPartitionedCall?
drop4/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_drop4_layer_call_and_return_conditional_losses_2032842
drop4/PartitionedCall?
up4_0/PartitionedCallPartitionedCalldrop4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_up4_0_layer_call_and_return_conditional_losses_2030402
up4_0/PartitionedCall?
up4/StatefulPartitionedCallStatefulPartitionedCallup4_0/PartitionedCall:output:0
up4_204201
up4_204203*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_up4_layer_call_and_return_conditional_losses_2032982
up4/StatefulPartitionedCall?
 zero_padding2d_1/PartitionedCallPartitionedCall$up4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_2030532"
 zero_padding2d_1/PartitionedCall?
concat_3/PartitionedCallPartitionedCalldrop3/PartitionedCall:output:0)zero_padding2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_concat_3_layer_call_and_return_conditional_losses_2033122
concat_3/PartitionedCall?
"conv3_up_0/StatefulPartitionedCallStatefulPartitionedCall!concat_3/PartitionedCall:output:0conv3_up_0_204208conv3_up_0_204210*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv3_up_0_layer_call_and_return_conditional_losses_2033252$
"conv3_up_0/StatefulPartitionedCall?
 conv3_up/StatefulPartitionedCallStatefulPartitionedCall+conv3_up_0/StatefulPartitionedCall:output:0conv3_up_204213conv3_up_204215*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv3_up_layer_call_and_return_conditional_losses_2033422"
 conv3_up/StatefulPartitionedCall?
up3_0/PartitionedCallPartitionedCall)conv3_up/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_up3_0_layer_call_and_return_conditional_losses_2030722
up3_0/PartitionedCall?
up3/StatefulPartitionedCallStatefulPartitionedCallup3_0/PartitionedCall:output:0
up3_204219
up3_204221*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_up3_layer_call_and_return_conditional_losses_2033602
up3/StatefulPartitionedCall?
 zero_padding2d_2/PartitionedCallPartitionedCall$up3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_zero_padding2d_2_layer_call_and_return_conditional_losses_2030852"
 zero_padding2d_2/PartitionedCall?
concat_2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0)zero_padding2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????(#?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_concat_2_layer_call_and_return_conditional_losses_2033742
concat_2/PartitionedCall?
"conv2_up_0/StatefulPartitionedCallStatefulPartitionedCall!concat_2/PartitionedCall:output:0conv2_up_0_204226conv2_up_0_204228*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(#@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2_up_0_layer_call_and_return_conditional_losses_2033872$
"conv2_up_0/StatefulPartitionedCall?
 conv2_up/StatefulPartitionedCallStatefulPartitionedCall+conv2_up_0/StatefulPartitionedCall:output:0conv2_up_204231conv2_up_204233*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(#@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2_up_layer_call_and_return_conditional_losses_2034042"
 conv2_up/StatefulPartitionedCall?
up2_0/PartitionedCallPartitionedCall)conv2_up/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_up2_0_layer_call_and_return_conditional_losses_2031042
up2_0/PartitionedCall?
up2/StatefulPartitionedCallStatefulPartitionedCallup2_0/PartitionedCall:output:0
up2_204237
up2_204239*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_up2_layer_call_and_return_conditional_losses_2034222
up2/StatefulPartitionedCall?
concat_1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0$up2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_concat_1_layer_call_and_return_conditional_losses_2034352
concat_1/PartitionedCall?
"conv1_up_0/StatefulPartitionedCallStatefulPartitionedCall!concat_1/PartitionedCall:output:0conv1_up_0_204243conv1_up_0_204245*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv1_up_0_layer_call_and_return_conditional_losses_2034482$
"conv1_up_0/StatefulPartitionedCall?
 conv1_up/StatefulPartitionedCallStatefulPartitionedCall+conv1_up_0/StatefulPartitionedCall:output:0conv1_up_204248conv1_up_204250*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv1_up_layer_call_and_return_conditional_losses_2034652"
 conv1_up/StatefulPartitionedCall?
conv1_1/StatefulPartitionedCallStatefulPartitionedCall)conv1_up/StatefulPartitionedCall:output:0conv1_1_204253conv1_1_204255*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????PF*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv1_1_layer_call_and_return_conditional_losses_2034812!
conv1_1/StatefulPartitionedCall?
cropping2d/PartitionedCallPartitionedCall(conv1_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????OE* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_cropping2d_layer_call_and_return_conditional_losses_2031192
cropping2d/PartitionedCall?
IdentityIdentity#cropping2d/PartitionedCall:output:0^conv1/StatefulPartitionedCall ^conv1_0/StatefulPartitionedCall ^conv1_1/StatefulPartitionedCall!^conv1_up/StatefulPartitionedCall#^conv1_up_0/StatefulPartitionedCall^conv2/StatefulPartitionedCall ^conv2_0/StatefulPartitionedCall!^conv2_up/StatefulPartitionedCall#^conv2_up_0/StatefulPartitionedCall^conv3/StatefulPartitionedCall ^conv3_0/StatefulPartitionedCall!^conv3_up/StatefulPartitionedCall#^conv3_up_0/StatefulPartitionedCall^conv4/StatefulPartitionedCall ^conv4_0/StatefulPartitionedCall^up2/StatefulPartitionedCall^up3/StatefulPartitionedCall^up4/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????OE2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????OE: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2B
conv1_0/StatefulPartitionedCallconv1_0/StatefulPartitionedCall2B
conv1_1/StatefulPartitionedCallconv1_1/StatefulPartitionedCall2D
 conv1_up/StatefulPartitionedCall conv1_up/StatefulPartitionedCall2H
"conv1_up_0/StatefulPartitionedCall"conv1_up_0/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2B
conv2_0/StatefulPartitionedCallconv2_0/StatefulPartitionedCall2D
 conv2_up/StatefulPartitionedCall conv2_up/StatefulPartitionedCall2H
"conv2_up_0/StatefulPartitionedCall"conv2_up_0/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2B
conv3_0/StatefulPartitionedCallconv3_0/StatefulPartitionedCall2D
 conv3_up/StatefulPartitionedCall conv3_up/StatefulPartitionedCall2H
"conv3_up_0/StatefulPartitionedCall"conv3_up_0/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2B
conv4_0/StatefulPartitionedCallconv4_0/StatefulPartitionedCall2:
up2/StatefulPartitionedCallup2/StatefulPartitionedCall2:
up3/StatefulPartitionedCallup3/StatefulPartitionedCall2:
up4/StatefulPartitionedCallup4/StatefulPartitionedCall:X T
/
_output_shapes
:?????????OE
!
_user_specified_name	input_1
?
?
A__inference_conv4_layer_call_and_return_conditional_losses_203273

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????
?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????
?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????
?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?
?
F__inference_conv2_up_0_layer_call_and_return_conditional_losses_203387

inputs9
conv2d_readvariableop_resource:?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(#@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????(#@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????(#?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????(#?
 
_user_specified_nameinputs
?
?
&__inference_conv3_layer_call_fn_205055

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv3_layer_call_and_return_conditional_losses_2032312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_conv3_0_layer_call_fn_205035

inputs"
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv3_0_layer_call_and_return_conditional_losses_2032142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
D__inference_conv2_up_layer_call_and_return_conditional_losses_205306

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(#@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????(#@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(#@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????(#@
 
_user_specified_nameinputs
?
?
D__inference_conv3_up_layer_call_and_return_conditional_losses_203342

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
B
&__inference_pool1_layer_call_fn_203003

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_pool1_layer_call_and_return_conditional_losses_2029972
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
M
1__inference_zero_padding2d_1_layer_call_fn_203059

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_2030532
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
B
&__inference_up2_0_layer_call_fn_203110

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_up2_0_layer_call_and_return_conditional_losses_2031042
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_202978
input_1F
,model_conv1_0_conv2d_readvariableop_resource: ;
-model_conv1_0_biasadd_readvariableop_resource: D
*model_conv1_conv2d_readvariableop_resource:  9
+model_conv1_biasadd_readvariableop_resource: F
,model_conv2_0_conv2d_readvariableop_resource: @;
-model_conv2_0_biasadd_readvariableop_resource:@D
*model_conv2_conv2d_readvariableop_resource:@@9
+model_conv2_biasadd_readvariableop_resource:@G
,model_conv3_0_conv2d_readvariableop_resource:@?<
-model_conv3_0_biasadd_readvariableop_resource:	?F
*model_conv3_conv2d_readvariableop_resource:??:
+model_conv3_biasadd_readvariableop_resource:	?H
,model_conv4_0_conv2d_readvariableop_resource:??<
-model_conv4_0_biasadd_readvariableop_resource:	?F
*model_conv4_conv2d_readvariableop_resource:??:
+model_conv4_biasadd_readvariableop_resource:	?D
(model_up4_conv2d_readvariableop_resource:??8
)model_up4_biasadd_readvariableop_resource:	?K
/model_conv3_up_0_conv2d_readvariableop_resource:???
0model_conv3_up_0_biasadd_readvariableop_resource:	?I
-model_conv3_up_conv2d_readvariableop_resource:??=
.model_conv3_up_biasadd_readvariableop_resource:	?C
(model_up3_conv2d_readvariableop_resource:?@7
)model_up3_biasadd_readvariableop_resource:@J
/model_conv2_up_0_conv2d_readvariableop_resource:?@>
0model_conv2_up_0_biasadd_readvariableop_resource:@G
-model_conv2_up_conv2d_readvariableop_resource:@@<
.model_conv2_up_biasadd_readvariableop_resource:@B
(model_up2_conv2d_readvariableop_resource:@ 7
)model_up2_biasadd_readvariableop_resource: I
/model_conv1_up_0_conv2d_readvariableop_resource:@ >
0model_conv1_up_0_biasadd_readvariableop_resource: G
-model_conv1_up_conv2d_readvariableop_resource:  <
.model_conv1_up_biasadd_readvariableop_resource: F
,model_conv1_1_conv2d_readvariableop_resource: ;
-model_conv1_1_biasadd_readvariableop_resource:
identity??"model/conv1/BiasAdd/ReadVariableOp?!model/conv1/Conv2D/ReadVariableOp?$model/conv1_0/BiasAdd/ReadVariableOp?#model/conv1_0/Conv2D/ReadVariableOp?$model/conv1_1/BiasAdd/ReadVariableOp?#model/conv1_1/Conv2D/ReadVariableOp?%model/conv1_up/BiasAdd/ReadVariableOp?$model/conv1_up/Conv2D/ReadVariableOp?'model/conv1_up_0/BiasAdd/ReadVariableOp?&model/conv1_up_0/Conv2D/ReadVariableOp?"model/conv2/BiasAdd/ReadVariableOp?!model/conv2/Conv2D/ReadVariableOp?$model/conv2_0/BiasAdd/ReadVariableOp?#model/conv2_0/Conv2D/ReadVariableOp?%model/conv2_up/BiasAdd/ReadVariableOp?$model/conv2_up/Conv2D/ReadVariableOp?'model/conv2_up_0/BiasAdd/ReadVariableOp?&model/conv2_up_0/Conv2D/ReadVariableOp?"model/conv3/BiasAdd/ReadVariableOp?!model/conv3/Conv2D/ReadVariableOp?$model/conv3_0/BiasAdd/ReadVariableOp?#model/conv3_0/Conv2D/ReadVariableOp?%model/conv3_up/BiasAdd/ReadVariableOp?$model/conv3_up/Conv2D/ReadVariableOp?'model/conv3_up_0/BiasAdd/ReadVariableOp?&model/conv3_up_0/Conv2D/ReadVariableOp?"model/conv4/BiasAdd/ReadVariableOp?!model/conv4/Conv2D/ReadVariableOp?$model/conv4_0/BiasAdd/ReadVariableOp?#model/conv4_0/Conv2D/ReadVariableOp? model/up2/BiasAdd/ReadVariableOp?model/up2/Conv2D/ReadVariableOp? model/up3/BiasAdd/ReadVariableOp?model/up3/Conv2D/ReadVariableOp? model/up4/BiasAdd/ReadVariableOp?model/up4/Conv2D/ReadVariableOp?
!model/zero_padding2d/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               2#
!model/zero_padding2d/Pad/paddings?
model/zero_padding2d/PadPadinput_1*model/zero_padding2d/Pad/paddings:output:0*
T0*/
_output_shapes
:?????????PF2
model/zero_padding2d/Pad?
#model/conv1_0/Conv2D/ReadVariableOpReadVariableOp,model_conv1_0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02%
#model/conv1_0/Conv2D/ReadVariableOp?
model/conv1_0/Conv2DConv2D!model/zero_padding2d/Pad:output:0+model/conv1_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF *
paddingSAME*
strides
2
model/conv1_0/Conv2D?
$model/conv1_0/BiasAdd/ReadVariableOpReadVariableOp-model_conv1_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$model/conv1_0/BiasAdd/ReadVariableOp?
model/conv1_0/BiasAddBiasAddmodel/conv1_0/Conv2D:output:0,model/conv1_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF 2
model/conv1_0/BiasAdd?
model/conv1_0/ReluRelumodel/conv1_0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????PF 2
model/conv1_0/Relu?
!model/conv1/Conv2D/ReadVariableOpReadVariableOp*model_conv1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02#
!model/conv1/Conv2D/ReadVariableOp?
model/conv1/Conv2DConv2D model/conv1_0/Relu:activations:0)model/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF *
paddingSAME*
strides
2
model/conv1/Conv2D?
"model/conv1/BiasAdd/ReadVariableOpReadVariableOp+model_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"model/conv1/BiasAdd/ReadVariableOp?
model/conv1/BiasAddBiasAddmodel/conv1/Conv2D:output:0*model/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF 2
model/conv1/BiasAdd?
model/conv1/ReluRelumodel/conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????PF 2
model/conv1/Relu?
model/pool1/MaxPoolMaxPoolmodel/conv1/Relu:activations:0*/
_output_shapes
:?????????(# *
ksize
*
paddingVALID*
strides
2
model/pool1/MaxPool?
#model/conv2_0/Conv2D/ReadVariableOpReadVariableOp,model_conv2_0_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02%
#model/conv2_0/Conv2D/ReadVariableOp?
model/conv2_0/Conv2DConv2Dmodel/pool1/MaxPool:output:0+model/conv2_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@*
paddingSAME*
strides
2
model/conv2_0/Conv2D?
$model/conv2_0/BiasAdd/ReadVariableOpReadVariableOp-model_conv2_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$model/conv2_0/BiasAdd/ReadVariableOp?
model/conv2_0/BiasAddBiasAddmodel/conv2_0/Conv2D:output:0,model/conv2_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@2
model/conv2_0/BiasAdd?
model/conv2_0/ReluRelumodel/conv2_0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(#@2
model/conv2_0/Relu?
!model/conv2/Conv2D/ReadVariableOpReadVariableOp*model_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02#
!model/conv2/Conv2D/ReadVariableOp?
model/conv2/Conv2DConv2D model/conv2_0/Relu:activations:0)model/conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@*
paddingSAME*
strides
2
model/conv2/Conv2D?
"model/conv2/BiasAdd/ReadVariableOpReadVariableOp+model_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"model/conv2/BiasAdd/ReadVariableOp?
model/conv2/BiasAddBiasAddmodel/conv2/Conv2D:output:0*model/conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@2
model/conv2/BiasAdd?
model/conv2/ReluRelumodel/conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(#@2
model/conv2/Relu?
model/pool2/MaxPoolMaxPoolmodel/conv2/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
model/pool2/MaxPool?
#model/conv3_0/Conv2D/ReadVariableOpReadVariableOp,model_conv3_0_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02%
#model/conv3_0/Conv2D/ReadVariableOp?
model/conv3_0/Conv2DConv2Dmodel/pool2/MaxPool:output:0+model/conv3_0/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model/conv3_0/Conv2D?
$model/conv3_0/BiasAdd/ReadVariableOpReadVariableOp-model_conv3_0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$model/conv3_0/BiasAdd/ReadVariableOp?
model/conv3_0/BiasAddBiasAddmodel/conv3_0/Conv2D:output:0,model/conv3_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model/conv3_0/BiasAdd?
model/conv3_0/ReluRelumodel/conv3_0/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/conv3_0/Relu?
!model/conv3/Conv2D/ReadVariableOpReadVariableOp*model_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02#
!model/conv3/Conv2D/ReadVariableOp?
model/conv3/Conv2DConv2D model/conv3_0/Relu:activations:0)model/conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model/conv3/Conv2D?
"model/conv3/BiasAdd/ReadVariableOpReadVariableOp+model_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/conv3/BiasAdd/ReadVariableOp?
model/conv3/BiasAddBiasAddmodel/conv3/Conv2D:output:0*model/conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model/conv3/BiasAdd?
model/conv3/ReluRelumodel/conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/conv3/Relu?
model/drop3/IdentityIdentitymodel/conv3/Relu:activations:0*
T0*0
_output_shapes
:??????????2
model/drop3/Identity?
model/pool3/MaxPoolMaxPoolmodel/drop3/Identity:output:0*0
_output_shapes
:?????????
?*
ksize
*
paddingVALID*
strides
2
model/pool3/MaxPool?
#model/conv4_0/Conv2D/ReadVariableOpReadVariableOp,model_conv4_0_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02%
#model/conv4_0/Conv2D/ReadVariableOp?
model/conv4_0/Conv2DConv2Dmodel/pool3/MaxPool:output:0+model/conv4_0/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?*
paddingSAME*
strides
2
model/conv4_0/Conv2D?
$model/conv4_0/BiasAdd/ReadVariableOpReadVariableOp-model_conv4_0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$model/conv4_0/BiasAdd/ReadVariableOp?
model/conv4_0/BiasAddBiasAddmodel/conv4_0/Conv2D:output:0,model/conv4_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?2
model/conv4_0/BiasAdd?
model/conv4_0/ReluRelumodel/conv4_0/BiasAdd:output:0*
T0*0
_output_shapes
:?????????
?2
model/conv4_0/Relu?
!model/conv4/Conv2D/ReadVariableOpReadVariableOp*model_conv4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02#
!model/conv4/Conv2D/ReadVariableOp?
model/conv4/Conv2DConv2D model/conv4_0/Relu:activations:0)model/conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?*
paddingSAME*
strides
2
model/conv4/Conv2D?
"model/conv4/BiasAdd/ReadVariableOpReadVariableOp+model_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/conv4/BiasAdd/ReadVariableOp?
model/conv4/BiasAddBiasAddmodel/conv4/Conv2D:output:0*model/conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?2
model/conv4/BiasAdd?
model/conv4/ReluRelumodel/conv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????
?2
model/conv4/Relu?
model/drop4/IdentityIdentitymodel/conv4/Relu:activations:0*
T0*0
_output_shapes
:?????????
?2
model/drop4/Identityw
model/up4_0/ConstConst*
_output_shapes
:*
dtype0*
valueB"
      2
model/up4_0/Const{
model/up4_0/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
model/up4_0/Const_1?
model/up4_0/mulMulmodel/up4_0/Const:output:0model/up4_0/Const_1:output:0*
T0*
_output_shapes
:2
model/up4_0/mul?
(model/up4_0/resize/ResizeNearestNeighborResizeNearestNeighbormodel/drop4/Identity:output:0model/up4_0/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2*
(model/up4_0/resize/ResizeNearestNeighbor?
model/up4/Conv2D/ReadVariableOpReadVariableOp(model_up4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
model/up4/Conv2D/ReadVariableOp?
model/up4/Conv2DConv2D9model/up4_0/resize/ResizeNearestNeighbor:resized_images:0'model/up4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model/up4/Conv2D?
 model/up4/BiasAdd/ReadVariableOpReadVariableOp)model_up4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 model/up4/BiasAdd/ReadVariableOp?
model/up4/BiasAddBiasAddmodel/up4/Conv2D:output:0(model/up4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model/up4/BiasAdd
model/up4/ReluRelumodel/up4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/up4/Relu?
#model/zero_padding2d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                2%
#model/zero_padding2d_1/Pad/paddings?
model/zero_padding2d_1/PadPadmodel/up4/Relu:activations:0,model/zero_padding2d_1/Pad/paddings:output:0*
T0*0
_output_shapes
:??????????2
model/zero_padding2d_1/Padz
model/concat_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concat_3/concat/axis?
model/concat_3/concatConcatV2model/drop3/Identity:output:0#model/zero_padding2d_1/Pad:output:0#model/concat_3/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
model/concat_3/concat?
&model/conv3_up_0/Conv2D/ReadVariableOpReadVariableOp/model_conv3_up_0_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&model/conv3_up_0/Conv2D/ReadVariableOp?
model/conv3_up_0/Conv2DConv2Dmodel/concat_3/concat:output:0.model/conv3_up_0/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model/conv3_up_0/Conv2D?
'model/conv3_up_0/BiasAdd/ReadVariableOpReadVariableOp0model_conv3_up_0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model/conv3_up_0/BiasAdd/ReadVariableOp?
model/conv3_up_0/BiasAddBiasAdd model/conv3_up_0/Conv2D:output:0/model/conv3_up_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model/conv3_up_0/BiasAdd?
model/conv3_up_0/ReluRelu!model/conv3_up_0/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/conv3_up_0/Relu?
$model/conv3_up/Conv2D/ReadVariableOpReadVariableOp-model_conv3_up_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$model/conv3_up/Conv2D/ReadVariableOp?
model/conv3_up/Conv2DConv2D#model/conv3_up_0/Relu:activations:0,model/conv3_up/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model/conv3_up/Conv2D?
%model/conv3_up/BiasAdd/ReadVariableOpReadVariableOp.model_conv3_up_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%model/conv3_up/BiasAdd/ReadVariableOp?
model/conv3_up/BiasAddBiasAddmodel/conv3_up/Conv2D:output:0-model/conv3_up/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model/conv3_up/BiasAdd?
model/conv3_up/ReluRelumodel/conv3_up/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/conv3_up/Reluw
model/up3_0/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model/up3_0/Const{
model/up3_0/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
model/up3_0/Const_1?
model/up3_0/mulMulmodel/up3_0/Const:output:0model/up3_0/Const_1:output:0*
T0*
_output_shapes
:2
model/up3_0/mul?
(model/up3_0/resize/ResizeNearestNeighborResizeNearestNeighbor!model/conv3_up/Relu:activations:0model/up3_0/mul:z:0*
T0*0
_output_shapes
:?????????("?*
half_pixel_centers(2*
(model/up3_0/resize/ResizeNearestNeighbor?
model/up3/Conv2D/ReadVariableOpReadVariableOp(model_up3_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02!
model/up3/Conv2D/ReadVariableOp?
model/up3/Conv2DConv2D9model/up3_0/resize/ResizeNearestNeighbor:resized_images:0'model/up3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????("@*
paddingSAME*
strides
2
model/up3/Conv2D?
 model/up3/BiasAdd/ReadVariableOpReadVariableOp)model_up3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 model/up3/BiasAdd/ReadVariableOp?
model/up3/BiasAddBiasAddmodel/up3/Conv2D:output:0(model/up3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????("@2
model/up3/BiasAdd~
model/up3/ReluRelumodel/up3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????("@2
model/up3/Relu?
#model/zero_padding2d_2/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                2%
#model/zero_padding2d_2/Pad/paddings?
model/zero_padding2d_2/PadPadmodel/up3/Relu:activations:0,model/zero_padding2d_2/Pad/paddings:output:0*
T0*/
_output_shapes
:?????????(#@2
model/zero_padding2d_2/Padz
model/concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concat_2/concat/axis?
model/concat_2/concatConcatV2model/conv2/Relu:activations:0#model/zero_padding2d_2/Pad:output:0#model/concat_2/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????(#?2
model/concat_2/concat?
&model/conv2_up_0/Conv2D/ReadVariableOpReadVariableOp/model_conv2_up_0_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02(
&model/conv2_up_0/Conv2D/ReadVariableOp?
model/conv2_up_0/Conv2DConv2Dmodel/concat_2/concat:output:0.model/conv2_up_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@*
paddingSAME*
strides
2
model/conv2_up_0/Conv2D?
'model/conv2_up_0/BiasAdd/ReadVariableOpReadVariableOp0model_conv2_up_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model/conv2_up_0/BiasAdd/ReadVariableOp?
model/conv2_up_0/BiasAddBiasAdd model/conv2_up_0/Conv2D:output:0/model/conv2_up_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@2
model/conv2_up_0/BiasAdd?
model/conv2_up_0/ReluRelu!model/conv2_up_0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(#@2
model/conv2_up_0/Relu?
$model/conv2_up/Conv2D/ReadVariableOpReadVariableOp-model_conv2_up_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02&
$model/conv2_up/Conv2D/ReadVariableOp?
model/conv2_up/Conv2DConv2D#model/conv2_up_0/Relu:activations:0,model/conv2_up/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@*
paddingSAME*
strides
2
model/conv2_up/Conv2D?
%model/conv2_up/BiasAdd/ReadVariableOpReadVariableOp.model_conv2_up_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2_up/BiasAdd/ReadVariableOp?
model/conv2_up/BiasAddBiasAddmodel/conv2_up/Conv2D:output:0-model/conv2_up/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(#@2
model/conv2_up/BiasAdd?
model/conv2_up/ReluRelumodel/conv2_up/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(#@2
model/conv2_up/Reluw
model/up2_0/ConstConst*
_output_shapes
:*
dtype0*
valueB"(   #   2
model/up2_0/Const{
model/up2_0/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
model/up2_0/Const_1?
model/up2_0/mulMulmodel/up2_0/Const:output:0model/up2_0/Const_1:output:0*
T0*
_output_shapes
:2
model/up2_0/mul?
(model/up2_0/resize/ResizeNearestNeighborResizeNearestNeighbor!model/conv2_up/Relu:activations:0model/up2_0/mul:z:0*
T0*/
_output_shapes
:?????????PF@*
half_pixel_centers(2*
(model/up2_0/resize/ResizeNearestNeighbor?
model/up2/Conv2D/ReadVariableOpReadVariableOp(model_up2_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
model/up2/Conv2D/ReadVariableOp?
model/up2/Conv2DConv2D9model/up2_0/resize/ResizeNearestNeighbor:resized_images:0'model/up2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF *
paddingSAME*
strides
2
model/up2/Conv2D?
 model/up2/BiasAdd/ReadVariableOpReadVariableOp)model_up2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 model/up2/BiasAdd/ReadVariableOp?
model/up2/BiasAddBiasAddmodel/up2/Conv2D:output:0(model/up2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF 2
model/up2/BiasAdd~
model/up2/ReluRelumodel/up2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????PF 2
model/up2/Reluz
model/concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concat_1/concat/axis?
model/concat_1/concatConcatV2model/conv1/Relu:activations:0model/up2/Relu:activations:0#model/concat_1/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????PF@2
model/concat_1/concat?
&model/conv1_up_0/Conv2D/ReadVariableOpReadVariableOp/model_conv1_up_0_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02(
&model/conv1_up_0/Conv2D/ReadVariableOp?
model/conv1_up_0/Conv2DConv2Dmodel/concat_1/concat:output:0.model/conv1_up_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF *
paddingSAME*
strides
2
model/conv1_up_0/Conv2D?
'model/conv1_up_0/BiasAdd/ReadVariableOpReadVariableOp0model_conv1_up_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model/conv1_up_0/BiasAdd/ReadVariableOp?
model/conv1_up_0/BiasAddBiasAdd model/conv1_up_0/Conv2D:output:0/model/conv1_up_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF 2
model/conv1_up_0/BiasAdd?
model/conv1_up_0/ReluRelu!model/conv1_up_0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????PF 2
model/conv1_up_0/Relu?
$model/conv1_up/Conv2D/ReadVariableOpReadVariableOp-model_conv1_up_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02&
$model/conv1_up/Conv2D/ReadVariableOp?
model/conv1_up/Conv2DConv2D#model/conv1_up_0/Relu:activations:0,model/conv1_up/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF *
paddingSAME*
strides
2
model/conv1_up/Conv2D?
%model/conv1_up/BiasAdd/ReadVariableOpReadVariableOp.model_conv1_up_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%model/conv1_up/BiasAdd/ReadVariableOp?
model/conv1_up/BiasAddBiasAddmodel/conv1_up/Conv2D:output:0-model/conv1_up/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF 2
model/conv1_up/BiasAdd?
model/conv1_up/ReluRelumodel/conv1_up/BiasAdd:output:0*
T0*/
_output_shapes
:?????????PF 2
model/conv1_up/Relu?
#model/conv1_1/Conv2D/ReadVariableOpReadVariableOp,model_conv1_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02%
#model/conv1_1/Conv2D/ReadVariableOp?
model/conv1_1/Conv2DConv2D!model/conv1_up/Relu:activations:0+model/conv1_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF*
paddingVALID*
strides
2
model/conv1_1/Conv2D?
$model/conv1_1/BiasAdd/ReadVariableOpReadVariableOp-model_conv1_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/conv1_1/BiasAdd/ReadVariableOp?
model/conv1_1/BiasAddBiasAddmodel/conv1_1/Conv2D:output:0,model/conv1_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF2
model/conv1_1/BiasAdd?
$model/cropping2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2&
$model/cropping2d/strided_slice/stack?
&model/cropping2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"    ????????    2(
&model/cropping2d/strided_slice/stack_1?
&model/cropping2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2(
&model/cropping2d/strided_slice/stack_2?
model/cropping2d/strided_sliceStridedSlicemodel/conv1_1/BiasAdd:output:0-model/cropping2d/strided_slice/stack:output:0/model/cropping2d/strided_slice/stack_1:output:0/model/cropping2d/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????OE*

begin_mask	*
end_mask	2 
model/cropping2d/strided_slice?
IdentityIdentity'model/cropping2d/strided_slice:output:0#^model/conv1/BiasAdd/ReadVariableOp"^model/conv1/Conv2D/ReadVariableOp%^model/conv1_0/BiasAdd/ReadVariableOp$^model/conv1_0/Conv2D/ReadVariableOp%^model/conv1_1/BiasAdd/ReadVariableOp$^model/conv1_1/Conv2D/ReadVariableOp&^model/conv1_up/BiasAdd/ReadVariableOp%^model/conv1_up/Conv2D/ReadVariableOp(^model/conv1_up_0/BiasAdd/ReadVariableOp'^model/conv1_up_0/Conv2D/ReadVariableOp#^model/conv2/BiasAdd/ReadVariableOp"^model/conv2/Conv2D/ReadVariableOp%^model/conv2_0/BiasAdd/ReadVariableOp$^model/conv2_0/Conv2D/ReadVariableOp&^model/conv2_up/BiasAdd/ReadVariableOp%^model/conv2_up/Conv2D/ReadVariableOp(^model/conv2_up_0/BiasAdd/ReadVariableOp'^model/conv2_up_0/Conv2D/ReadVariableOp#^model/conv3/BiasAdd/ReadVariableOp"^model/conv3/Conv2D/ReadVariableOp%^model/conv3_0/BiasAdd/ReadVariableOp$^model/conv3_0/Conv2D/ReadVariableOp&^model/conv3_up/BiasAdd/ReadVariableOp%^model/conv3_up/Conv2D/ReadVariableOp(^model/conv3_up_0/BiasAdd/ReadVariableOp'^model/conv3_up_0/Conv2D/ReadVariableOp#^model/conv4/BiasAdd/ReadVariableOp"^model/conv4/Conv2D/ReadVariableOp%^model/conv4_0/BiasAdd/ReadVariableOp$^model/conv4_0/Conv2D/ReadVariableOp!^model/up2/BiasAdd/ReadVariableOp ^model/up2/Conv2D/ReadVariableOp!^model/up3/BiasAdd/ReadVariableOp ^model/up3/Conv2D/ReadVariableOp!^model/up4/BiasAdd/ReadVariableOp ^model/up4/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????OE2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????OE: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"model/conv1/BiasAdd/ReadVariableOp"model/conv1/BiasAdd/ReadVariableOp2F
!model/conv1/Conv2D/ReadVariableOp!model/conv1/Conv2D/ReadVariableOp2L
$model/conv1_0/BiasAdd/ReadVariableOp$model/conv1_0/BiasAdd/ReadVariableOp2J
#model/conv1_0/Conv2D/ReadVariableOp#model/conv1_0/Conv2D/ReadVariableOp2L
$model/conv1_1/BiasAdd/ReadVariableOp$model/conv1_1/BiasAdd/ReadVariableOp2J
#model/conv1_1/Conv2D/ReadVariableOp#model/conv1_1/Conv2D/ReadVariableOp2N
%model/conv1_up/BiasAdd/ReadVariableOp%model/conv1_up/BiasAdd/ReadVariableOp2L
$model/conv1_up/Conv2D/ReadVariableOp$model/conv1_up/Conv2D/ReadVariableOp2R
'model/conv1_up_0/BiasAdd/ReadVariableOp'model/conv1_up_0/BiasAdd/ReadVariableOp2P
&model/conv1_up_0/Conv2D/ReadVariableOp&model/conv1_up_0/Conv2D/ReadVariableOp2H
"model/conv2/BiasAdd/ReadVariableOp"model/conv2/BiasAdd/ReadVariableOp2F
!model/conv2/Conv2D/ReadVariableOp!model/conv2/Conv2D/ReadVariableOp2L
$model/conv2_0/BiasAdd/ReadVariableOp$model/conv2_0/BiasAdd/ReadVariableOp2J
#model/conv2_0/Conv2D/ReadVariableOp#model/conv2_0/Conv2D/ReadVariableOp2N
%model/conv2_up/BiasAdd/ReadVariableOp%model/conv2_up/BiasAdd/ReadVariableOp2L
$model/conv2_up/Conv2D/ReadVariableOp$model/conv2_up/Conv2D/ReadVariableOp2R
'model/conv2_up_0/BiasAdd/ReadVariableOp'model/conv2_up_0/BiasAdd/ReadVariableOp2P
&model/conv2_up_0/Conv2D/ReadVariableOp&model/conv2_up_0/Conv2D/ReadVariableOp2H
"model/conv3/BiasAdd/ReadVariableOp"model/conv3/BiasAdd/ReadVariableOp2F
!model/conv3/Conv2D/ReadVariableOp!model/conv3/Conv2D/ReadVariableOp2L
$model/conv3_0/BiasAdd/ReadVariableOp$model/conv3_0/BiasAdd/ReadVariableOp2J
#model/conv3_0/Conv2D/ReadVariableOp#model/conv3_0/Conv2D/ReadVariableOp2N
%model/conv3_up/BiasAdd/ReadVariableOp%model/conv3_up/BiasAdd/ReadVariableOp2L
$model/conv3_up/Conv2D/ReadVariableOp$model/conv3_up/Conv2D/ReadVariableOp2R
'model/conv3_up_0/BiasAdd/ReadVariableOp'model/conv3_up_0/BiasAdd/ReadVariableOp2P
&model/conv3_up_0/Conv2D/ReadVariableOp&model/conv3_up_0/Conv2D/ReadVariableOp2H
"model/conv4/BiasAdd/ReadVariableOp"model/conv4/BiasAdd/ReadVariableOp2F
!model/conv4/Conv2D/ReadVariableOp!model/conv4/Conv2D/ReadVariableOp2L
$model/conv4_0/BiasAdd/ReadVariableOp$model/conv4_0/BiasAdd/ReadVariableOp2J
#model/conv4_0/Conv2D/ReadVariableOp#model/conv4_0/Conv2D/ReadVariableOp2D
 model/up2/BiasAdd/ReadVariableOp model/up2/BiasAdd/ReadVariableOp2B
model/up2/Conv2D/ReadVariableOpmodel/up2/Conv2D/ReadVariableOp2D
 model/up3/BiasAdd/ReadVariableOp model/up3/BiasAdd/ReadVariableOp2B
model/up3/Conv2D/ReadVariableOpmodel/up3/Conv2D/ReadVariableOp2D
 model/up4/BiasAdd/ReadVariableOp model/up4/BiasAdd/ReadVariableOp2B
model/up4/Conv2D/ReadVariableOpmodel/up4/Conv2D/ReadVariableOp:X T
/
_output_shapes
:?????????OE
!
_user_specified_name	input_1
?
h
L__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_203053

inputs
identity?
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                2
Pad/paddings?
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Pad?
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_zero_padding2d_layer_call_and_return_conditional_losses_202985

inputs
identity?
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                               2
Pad/paddings?
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Pad?
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
B
&__inference_drop3_layer_call_fn_205071

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_drop3_layer_call_and_return_conditional_losses_2032422
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
A__inference_drop4_layer_call_and_return_conditional_losses_205148

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????
?2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????
?2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????
?:X T
0
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?
?
(__inference_conv2_0_layer_call_fn_204995

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(#@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2_0_layer_call_and_return_conditional_losses_2031792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????(#@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(# : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(# 
 
_user_specified_nameinputs
?
B
&__inference_up3_0_layer_call_fn_203078

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_up3_0_layer_call_and_return_conditional_losses_2030722
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_conv1_up_layer_call_and_return_conditional_losses_205379

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????PF 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????PF 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????PF : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????PF 
 
_user_specified_nameinputs
?
?
A__inference_conv3_layer_call_and_return_conditional_losses_203231

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
n
D__inference_concat_1_layer_call_and_return_conditional_losses_203435

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????PF@2
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:?????????PF@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:?????????PF :+??????????????????????????? :W S
/
_output_shapes
:?????????PF 
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?	
&__inference_model_layer_call_fn_203564
input_1!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?%

unknown_21:?@

unknown_22:@%

unknown_23:?@

unknown_24:@$

unknown_25:@@

unknown_26:@$

unknown_27:@ 

unknown_28: $

unknown_29:@ 

unknown_30: $

unknown_31:  

unknown_32: $

unknown_33: 

unknown_34:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????OE*F
_read_only_resource_inputs(
&$	
 !"#$*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2034892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????OE2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????OE: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????OE
!
_user_specified_name	input_1
?
?
$__inference_up3_layer_call_fn_205242

inputs"
unknown:?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_up3_layer_call_and_return_conditional_losses_2033602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_conv2_up_0_layer_call_fn_205275

inputs"
unknown:?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(#@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_conv2_up_0_layer_call_and_return_conditional_losses_2033872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????(#@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????(#?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????(#?
 
_user_specified_nameinputs
?
?
F__inference_conv3_up_0_layer_call_and_return_conditional_losses_203325

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
A__inference_drop3_layer_call_and_return_conditional_losses_205081

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_conv3_up_0_layer_call_and_return_conditional_losses_205213

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
B
&__inference_pool2_layer_call_fn_203015

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_pool2_layer_call_and_return_conditional_losses_2030092
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
n
D__inference_concat_3_layer_call_and_return_conditional_losses_203312

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:??????????:,????????????????????????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs:jf
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_conv2_layer_call_fn_205015

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(#@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_2031962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????(#@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(#@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(#@
 
_user_specified_nameinputs
?
?
A__inference_conv1_layer_call_and_return_conditional_losses_204986

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????PF 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????PF 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????PF : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????PF 
 
_user_specified_nameinputs
?
?
C__inference_conv3_0_layer_call_and_return_conditional_losses_205046

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
B
&__inference_drop4_layer_call_fn_205138

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_drop4_layer_call_and_return_conditional_losses_2032842
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????
?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????
?:X T
0
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?
`
A__inference_drop3_layer_call_and_return_conditional_losses_203748

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
A__inference_drop4_layer_call_and_return_conditional_losses_203284

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????
?2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????
?2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????
?:X T
0
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?
?
?__inference_up3_layer_call_and_return_conditional_losses_205253

inputs9
conv2d_readvariableop_resource:?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
C__inference_conv1_1_layer_call_and_return_conditional_losses_205398

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????PF2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????PF : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????PF 
 
_user_specified_nameinputs
??
?4
"__inference__traced_restore_205929
file_prefix9
assignvariableop_conv1_0_kernel: -
assignvariableop_1_conv1_0_bias: 9
assignvariableop_2_conv1_kernel:  +
assignvariableop_3_conv1_bias: ;
!assignvariableop_4_conv2_0_kernel: @-
assignvariableop_5_conv2_0_bias:@9
assignvariableop_6_conv2_kernel:@@+
assignvariableop_7_conv2_bias:@<
!assignvariableop_8_conv3_0_kernel:@?.
assignvariableop_9_conv3_0_bias:	?<
 assignvariableop_10_conv3_kernel:??-
assignvariableop_11_conv3_bias:	?>
"assignvariableop_12_conv4_0_kernel:??/
 assignvariableop_13_conv4_0_bias:	?<
 assignvariableop_14_conv4_kernel:??-
assignvariableop_15_conv4_bias:	?:
assignvariableop_16_up4_kernel:??+
assignvariableop_17_up4_bias:	?A
%assignvariableop_18_conv3_up_0_kernel:??2
#assignvariableop_19_conv3_up_0_bias:	??
#assignvariableop_20_conv3_up_kernel:??0
!assignvariableop_21_conv3_up_bias:	?9
assignvariableop_22_up3_kernel:?@*
assignvariableop_23_up3_bias:@@
%assignvariableop_24_conv2_up_0_kernel:?@1
#assignvariableop_25_conv2_up_0_bias:@=
#assignvariableop_26_conv2_up_kernel:@@/
!assignvariableop_27_conv2_up_bias:@8
assignvariableop_28_up2_kernel:@ *
assignvariableop_29_up2_bias: ?
%assignvariableop_30_conv1_up_0_kernel:@ 1
#assignvariableop_31_conv1_up_0_bias: =
#assignvariableop_32_conv1_up_kernel:  /
!assignvariableop_33_conv1_up_bias: <
"assignvariableop_34_conv1_1_kernel: .
 assignvariableop_35_conv1_1_bias:*
 assignvariableop_36_rmsprop_iter:	 +
!assignvariableop_37_rmsprop_decay: 3
)assignvariableop_38_rmsprop_learning_rate: .
$assignvariableop_39_rmsprop_momentum: )
assignvariableop_40_rmsprop_rho: #
assignvariableop_41_total: #
assignvariableop_42_count: %
assignvariableop_43_total_1: %
assignvariableop_44_count_1: %
assignvariableop_45_total_2: %
assignvariableop_46_count_2: H
.assignvariableop_47_rmsprop_conv1_0_kernel_rms: :
,assignvariableop_48_rmsprop_conv1_0_bias_rms: F
,assignvariableop_49_rmsprop_conv1_kernel_rms:  8
*assignvariableop_50_rmsprop_conv1_bias_rms: H
.assignvariableop_51_rmsprop_conv2_0_kernel_rms: @:
,assignvariableop_52_rmsprop_conv2_0_bias_rms:@F
,assignvariableop_53_rmsprop_conv2_kernel_rms:@@8
*assignvariableop_54_rmsprop_conv2_bias_rms:@I
.assignvariableop_55_rmsprop_conv3_0_kernel_rms:@?;
,assignvariableop_56_rmsprop_conv3_0_bias_rms:	?H
,assignvariableop_57_rmsprop_conv3_kernel_rms:??9
*assignvariableop_58_rmsprop_conv3_bias_rms:	?J
.assignvariableop_59_rmsprop_conv4_0_kernel_rms:??;
,assignvariableop_60_rmsprop_conv4_0_bias_rms:	?H
,assignvariableop_61_rmsprop_conv4_kernel_rms:??9
*assignvariableop_62_rmsprop_conv4_bias_rms:	?F
*assignvariableop_63_rmsprop_up4_kernel_rms:??7
(assignvariableop_64_rmsprop_up4_bias_rms:	?M
1assignvariableop_65_rmsprop_conv3_up_0_kernel_rms:??>
/assignvariableop_66_rmsprop_conv3_up_0_bias_rms:	?K
/assignvariableop_67_rmsprop_conv3_up_kernel_rms:??<
-assignvariableop_68_rmsprop_conv3_up_bias_rms:	?E
*assignvariableop_69_rmsprop_up3_kernel_rms:?@6
(assignvariableop_70_rmsprop_up3_bias_rms:@L
1assignvariableop_71_rmsprop_conv2_up_0_kernel_rms:?@=
/assignvariableop_72_rmsprop_conv2_up_0_bias_rms:@I
/assignvariableop_73_rmsprop_conv2_up_kernel_rms:@@;
-assignvariableop_74_rmsprop_conv2_up_bias_rms:@D
*assignvariableop_75_rmsprop_up2_kernel_rms:@ 6
(assignvariableop_76_rmsprop_up2_bias_rms: K
1assignvariableop_77_rmsprop_conv1_up_0_kernel_rms:@ =
/assignvariableop_78_rmsprop_conv1_up_0_bias_rms: I
/assignvariableop_79_rmsprop_conv1_up_kernel_rms:  ;
-assignvariableop_80_rmsprop_conv1_up_bias_rms: H
.assignvariableop_81_rmsprop_conv1_1_kernel_rms: :
,assignvariableop_82_rmsprop_conv1_1_bias_rms:
identity_84??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_9?-
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*?,
value?,B?,TB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*?
value?B?TB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*b
dtypesX
V2T	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv1_0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_conv2_0_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv2_0_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_conv2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_conv3_0_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_conv3_0_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp assignvariableop_10_conv3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_conv3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv4_0_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp assignvariableop_13_conv4_0_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp assignvariableop_14_conv4_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_conv4_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_up4_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_up4_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp%assignvariableop_18_conv3_up_0_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp#assignvariableop_19_conv3_up_0_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp#assignvariableop_20_conv3_up_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp!assignvariableop_21_conv3_up_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_up3_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_up3_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp%assignvariableop_24_conv2_up_0_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp#assignvariableop_25_conv2_up_0_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp#assignvariableop_26_conv2_up_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp!assignvariableop_27_conv2_up_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_up2_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_up2_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp%assignvariableop_30_conv1_up_0_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp#assignvariableop_31_conv1_up_0_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp#assignvariableop_32_conv1_up_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp!assignvariableop_33_conv1_up_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp"assignvariableop_34_conv1_1_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp assignvariableop_35_conv1_1_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp assignvariableop_36_rmsprop_iterIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp!assignvariableop_37_rmsprop_decayIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp)assignvariableop_38_rmsprop_learning_rateIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp$assignvariableop_39_rmsprop_momentumIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpassignvariableop_40_rmsprop_rhoIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpassignvariableop_41_totalIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpassignvariableop_42_countIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpassignvariableop_43_total_1Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpassignvariableop_44_count_1Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpassignvariableop_45_total_2Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpassignvariableop_46_count_2Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp.assignvariableop_47_rmsprop_conv1_0_kernel_rmsIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp,assignvariableop_48_rmsprop_conv1_0_bias_rmsIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp,assignvariableop_49_rmsprop_conv1_kernel_rmsIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp*assignvariableop_50_rmsprop_conv1_bias_rmsIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp.assignvariableop_51_rmsprop_conv2_0_kernel_rmsIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp,assignvariableop_52_rmsprop_conv2_0_bias_rmsIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp,assignvariableop_53_rmsprop_conv2_kernel_rmsIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp*assignvariableop_54_rmsprop_conv2_bias_rmsIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp.assignvariableop_55_rmsprop_conv3_0_kernel_rmsIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp,assignvariableop_56_rmsprop_conv3_0_bias_rmsIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp,assignvariableop_57_rmsprop_conv3_kernel_rmsIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp*assignvariableop_58_rmsprop_conv3_bias_rmsIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp.assignvariableop_59_rmsprop_conv4_0_kernel_rmsIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp,assignvariableop_60_rmsprop_conv4_0_bias_rmsIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp,assignvariableop_61_rmsprop_conv4_kernel_rmsIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp*assignvariableop_62_rmsprop_conv4_bias_rmsIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp*assignvariableop_63_rmsprop_up4_kernel_rmsIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp(assignvariableop_64_rmsprop_up4_bias_rmsIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp1assignvariableop_65_rmsprop_conv3_up_0_kernel_rmsIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp/assignvariableop_66_rmsprop_conv3_up_0_bias_rmsIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp/assignvariableop_67_rmsprop_conv3_up_kernel_rmsIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp-assignvariableop_68_rmsprop_conv3_up_bias_rmsIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp*assignvariableop_69_rmsprop_up3_kernel_rmsIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp(assignvariableop_70_rmsprop_up3_bias_rmsIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp1assignvariableop_71_rmsprop_conv2_up_0_kernel_rmsIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp/assignvariableop_72_rmsprop_conv2_up_0_bias_rmsIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp/assignvariableop_73_rmsprop_conv2_up_kernel_rmsIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp-assignvariableop_74_rmsprop_conv2_up_bias_rmsIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp*assignvariableop_75_rmsprop_up2_kernel_rmsIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp(assignvariableop_76_rmsprop_up2_bias_rmsIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp1assignvariableop_77_rmsprop_conv1_up_0_kernel_rmsIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp/assignvariableop_78_rmsprop_conv1_up_0_bias_rmsIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp/assignvariableop_79_rmsprop_conv1_up_kernel_rmsIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp-assignvariableop_80_rmsprop_conv1_up_bias_rmsIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp.assignvariableop_81_rmsprop_conv1_1_kernel_rmsIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp,assignvariableop_82_rmsprop_conv1_1_bias_rmsIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_829
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_83Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_83?
Identity_84IdentityIdentity_83:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_84"#
identity_84Identity_84:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
)__inference_conv2_up_layer_call_fn_205295

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(#@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv2_up_layer_call_and_return_conditional_losses_2034042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????(#@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(#@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(#@
 
_user_specified_nameinputs
?
?
)__inference_conv3_up_layer_call_fn_205222

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_conv3_up_layer_call_and_return_conditional_losses_2033422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_conv1_up_0_layer_call_and_return_conditional_losses_203448

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????PF 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????PF 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????PF@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????PF@
 
_user_specified_nameinputs
?
?
F__inference_conv1_up_0_layer_call_and_return_conditional_losses_205359

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????PF 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????PF 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????PF 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????PF@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????PF@
 
_user_specified_nameinputs
?
?	
$__inference_signature_wrapper_204454
input_1!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?%

unknown_21:?@

unknown_22:@%

unknown_23:?@

unknown_24:@$

unknown_25:@@

unknown_26:@$

unknown_27:@ 

unknown_28: $

unknown_29:@ 

unknown_30: $

unknown_31:  

unknown_32: $

unknown_33: 

unknown_34:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????OE*F
_read_only_resource_inputs(
&$	
 !"#$*2
config_proto" 

CPU

GPU2 *0J 8? **
f%R#
!__inference__wrapped_model_2029782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????OE2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:?????????OE: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????OE
!
_user_specified_name	input_1
?
B
&__inference_up4_0_layer_call_fn_203046

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_up4_0_layer_call_and_return_conditional_losses_2030402
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
p
D__inference_concat_2_layer_call_and_return_conditional_losses_205266
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????(#?2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????(#?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:?????????(#@:+???????????????????????????@:Y U
/
_output_shapes
:?????????(#@
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+???????????????????????????@
"
_user_specified_name
inputs/1
?
?
?__inference_up4_layer_call_and_return_conditional_losses_203298

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_zero_padding2d_2_layer_call_and_return_conditional_losses_203085

inputs
identity?
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                2
Pad/paddings?
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Pad?
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
G
+__inference_cropping2d_layer_call_fn_203125

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_cropping2d_layer_call_and_return_conditional_losses_2031192
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
?__inference_up2_layer_call_and_return_conditional_losses_205326

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????OEF

cropping2d8
StatefulPartitionedCall:0?????????OEtensorflow/serving/predict:??
??
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer-14
layer-15
layer_with_weights-8
layer-16
layer-17
layer-18
layer_with_weights-9
layer-19
layer_with_weights-10
layer-20
layer-21
layer_with_weights-11
layer-22
layer-23
layer-24
layer_with_weights-12
layer-25
layer_with_weights-13
layer-26
layer-27
layer_with_weights-14
layer-28
layer-29
layer_with_weights-15
layer-30
 layer_with_weights-16
 layer-31
!layer_with_weights-17
!layer-32
"layer-33
#	optimizer
$regularization_losses
%trainable_variables
&	variables
'	keras_api
(
signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"??
_tf_keras_network??{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 79, 69, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 79, 69, 1]}, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 1]}, {"class_name": "__tuple__", "items": [0, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1_0", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_0", "inbound_nodes": [[["zero_padding2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["conv1_0", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool1", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_0", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_0", "inbound_nodes": [[["pool1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["conv2_0", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool2", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_0", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_0", "inbound_nodes": [[["pool2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3", "inbound_nodes": [[["conv3_0", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "drop3", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "drop3", "inbound_nodes": [[["conv3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "pool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool3", "inbound_nodes": [[["drop3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_0", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_0", "inbound_nodes": [[["pool3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4", "inbound_nodes": [[["conv4_0", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "drop4", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "drop4", "inbound_nodes": [[["conv4", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up4_0", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up4_0", "inbound_nodes": [[["drop4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "up4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "up4", "inbound_nodes": [[["up4_0", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 0]}, {"class_name": "__tuple__", "items": [0, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_1", "inbound_nodes": [[["up4", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concat_3", "trainable": true, "dtype": "float32", "axis": 3}, "name": "concat_3", "inbound_nodes": [[["drop3", 0, 0, {}], ["zero_padding2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_up_0", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_up_0", "inbound_nodes": [[["concat_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_up", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_up", "inbound_nodes": [[["conv3_up_0", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up3_0", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up3_0", "inbound_nodes": [[["conv3_up", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "up3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "up3", "inbound_nodes": [[["up3_0", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_2", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 0]}, {"class_name": "__tuple__", "items": [0, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_2", "inbound_nodes": [[["up3", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concat_2", "trainable": true, "dtype": "float32", "axis": 3}, "name": "concat_2", "inbound_nodes": [[["conv2", 0, 0, {}], ["zero_padding2d_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_up_0", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_up_0", "inbound_nodes": [[["concat_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_up", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_up", "inbound_nodes": [[["conv2_up_0", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up2_0", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up2_0", "inbound_nodes": [[["conv2_up", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "up2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "up2", "inbound_nodes": [[["up2_0", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concat_1", "trainable": true, "dtype": "float32", "axis": 3}, "name": "concat_1", "inbound_nodes": [[["conv1", 0, 0, {}], ["up2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1_up_0", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_up_0", "inbound_nodes": [[["concat_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1_up", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_up", "inbound_nodes": [[["conv1_up_0", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1_1", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_1", "inbound_nodes": [[["conv1_up", 0, 0, {}]]]}, {"class_name": "Cropping2D", "config": {"name": "cropping2d", "trainable": true, "dtype": "float32", "cropping": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 1]}, {"class_name": "__tuple__", "items": [0, 1]}]}, "data_format": "channels_last"}, "name": "cropping2d", "inbound_nodes": [[["conv1_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["cropping2d", 0, 0]]}, "shared_object_id": 53, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 79, 69, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 79, 69, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 79, 69, 1]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 79, 69, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 79, 69, 1]}, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 1]}, {"class_name": "__tuple__", "items": [0, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 1}, {"class_name": "Conv2D", "config": {"name": "conv1_0", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_0", "inbound_nodes": [[["zero_padding2d", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["conv1_0", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "MaxPooling2D", "config": {"name": "pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool1", "inbound_nodes": [[["conv1", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Conv2D", "config": {"name": "conv2_0", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_0", "inbound_nodes": [[["pool1", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["conv2_0", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "MaxPooling2D", "config": {"name": "pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool2", "inbound_nodes": [[["conv2", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Conv2D", "config": {"name": "conv3_0", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_0", "inbound_nodes": [[["pool2", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "Conv2D", "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3", "inbound_nodes": [[["conv3_0", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "Dropout", "config": {"name": "drop3", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "drop3", "inbound_nodes": [[["conv3", 0, 0, {}]]], "shared_object_id": 16}, {"class_name": "MaxPooling2D", "config": {"name": "pool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool3", "inbound_nodes": [[["drop3", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv4_0", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_0", "inbound_nodes": [[["pool3", 0, 0, {}]]], "shared_object_id": 19}, {"class_name": "Conv2D", "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4", "inbound_nodes": [[["conv4_0", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "Dropout", "config": {"name": "drop4", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "drop4", "inbound_nodes": [[["conv4", 0, 0, {}]]], "shared_object_id": 22}, {"class_name": "UpSampling2D", "config": {"name": "up4_0", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up4_0", "inbound_nodes": [[["drop4", 0, 0, {}]]], "shared_object_id": 23}, {"class_name": "Conv2D", "config": {"name": "up4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "up4", "inbound_nodes": [[["up4_0", 0, 0, {}]]], "shared_object_id": 25}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 0]}, {"class_name": "__tuple__", "items": [0, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_1", "inbound_nodes": [[["up4", 0, 0, {}]]], "shared_object_id": 26}, {"class_name": "Concatenate", "config": {"name": "concat_3", "trainable": true, "dtype": "float32", "axis": 3}, "name": "concat_3", "inbound_nodes": [[["drop3", 0, 0, {}], ["zero_padding2d_1", 0, 0, {}]]], "shared_object_id": 27}, {"class_name": "Conv2D", "config": {"name": "conv3_up_0", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_up_0", "inbound_nodes": [[["concat_3", 0, 0, {}]]], "shared_object_id": 29}, {"class_name": "Conv2D", "config": {"name": "conv3_up", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_up", "inbound_nodes": [[["conv3_up_0", 0, 0, {}]]], "shared_object_id": 31}, {"class_name": "UpSampling2D", "config": {"name": "up3_0", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up3_0", "inbound_nodes": [[["conv3_up", 0, 0, {}]]], "shared_object_id": 32}, {"class_name": "Conv2D", "config": {"name": "up3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "up3", "inbound_nodes": [[["up3_0", 0, 0, {}]]], "shared_object_id": 34}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_2", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 0]}, {"class_name": "__tuple__", "items": [0, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_2", "inbound_nodes": [[["up3", 0, 0, {}]]], "shared_object_id": 35}, {"class_name": "Concatenate", "config": {"name": "concat_2", "trainable": true, "dtype": "float32", "axis": 3}, "name": "concat_2", "inbound_nodes": [[["conv2", 0, 0, {}], ["zero_padding2d_2", 0, 0, {}]]], "shared_object_id": 36}, {"class_name": "Conv2D", "config": {"name": "conv2_up_0", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 37}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_up_0", "inbound_nodes": [[["concat_2", 0, 0, {}]]], "shared_object_id": 38}, {"class_name": "Conv2D", "config": {"name": "conv2_up", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 39}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_up", "inbound_nodes": [[["conv2_up_0", 0, 0, {}]]], "shared_object_id": 40}, {"class_name": "UpSampling2D", "config": {"name": "up2_0", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up2_0", "inbound_nodes": [[["conv2_up", 0, 0, {}]]], "shared_object_id": 41}, {"class_name": "Conv2D", "config": {"name": "up2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "up2", "inbound_nodes": [[["up2_0", 0, 0, {}]]], "shared_object_id": 43}, {"class_name": "Concatenate", "config": {"name": "concat_1", "trainable": true, "dtype": "float32", "axis": 3}, "name": "concat_1", "inbound_nodes": [[["conv1", 0, 0, {}], ["up2", 0, 0, {}]]], "shared_object_id": 44}, {"class_name": "Conv2D", "config": {"name": "conv1_up_0", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 45}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_up_0", "inbound_nodes": [[["concat_1", 0, 0, {}]]], "shared_object_id": 46}, {"class_name": "Conv2D", "config": {"name": "conv1_up", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 47}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_up", "inbound_nodes": [[["conv1_up_0", 0, 0, {}]]], "shared_object_id": 48}, {"class_name": "Conv2D", "config": {"name": "conv1_1", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 49}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 50}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_1", "inbound_nodes": [[["conv1_up", 0, 0, {}]]], "shared_object_id": 51}, {"class_name": "Cropping2D", "config": {"name": "cropping2d", "trainable": true, "dtype": "float32", "cropping": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 1]}, {"class_name": "__tuple__", "items": [0, 1]}]}, "data_format": "channels_last"}, "name": "cropping2d", "inbound_nodes": [[["conv1_1", 0, 0, {}]]], "shared_object_id": 52}], "input_layers": [["input_1", 0, 0]], "output_layers": [["cropping2d", 0, 0]]}}, "training_config": {"loss": "mse", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}, "shared_object_id": 55}, {"class_name": "MeanMetricWrapper", "config": {"name": "root_mse", "dtype": "float32", "fn": "root_mse"}, "shared_object_id": 56}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0001, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 79, 69, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 79, 69, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?
)regularization_losses
*trainable_variables
+	variables
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "zero_padding2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 79, 69, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 79, 69, 1]}, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 1]}, {"class_name": "__tuple__", "items": [0, 1]}]}, "data_format": "channels_last"}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 1, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 57}}
?


-kernel
.bias
/regularization_losses
0trainable_variables
1	variables
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "conv1_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv1_0", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["zero_padding2d", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 58}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80, 70, 1]}}
?


3kernel
4bias
5regularization_losses
6trainable_variables
7	variables
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv1_0", 0, 0, {}]]], "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 59}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80, 70, 32]}}
?
9regularization_losses
:trainable_variables
;	variables
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "pool1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv1", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 60}}
?


=kernel
>bias
?regularization_losses
@trainable_variables
A	variables
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "conv2_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2_0", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["pool1", 0, 0, {}]]], "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 61}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 35, 32]}}
?


Ckernel
Dbias
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2_0", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 62}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 35, 64]}}
?
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "pool2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv2", 0, 0, {}]]], "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 63}}
?


Mkernel
Nbias
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "conv3_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv3_0", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["pool2", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 64}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 17, 64]}}
?


Skernel
Tbias
Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv3_0", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 65}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 17, 128]}}
?
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "drop3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "drop3", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "inbound_nodes": [[["conv3", 0, 0, {}]]], "shared_object_id": 16}
?
]regularization_losses
^trainable_variables
_	variables
`	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "pool3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "pool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["drop3", 0, 0, {}]]], "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 66}}
?


akernel
bbias
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "conv4_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv4_0", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["pool3", 0, 0, {}]]], "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 67}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 8, 128]}}
?


gkernel
hbias
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "conv4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv4_0", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 68}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 8, 256]}}
?
mregularization_losses
ntrainable_variables
o	variables
p	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "drop4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "drop4", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "inbound_nodes": [[["conv4", 0, 0, {}]]], "shared_object_id": 22}
?
qregularization_losses
rtrainable_variables
s	variables
t	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "up4_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling2D", "config": {"name": "up4_0", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "inbound_nodes": [[["drop4", 0, 0, {}]]], "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 69}}
?


ukernel
vbias
wregularization_losses
xtrainable_variables
y	variables
z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "up4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "up4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["up4_0", 0, 0, {}]]], "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 70}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 16, 256]}}
?
{regularization_losses
|trainable_variables
}	variables
~	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "zero_padding2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 0]}, {"class_name": "__tuple__", "items": [0, 1]}]}, "data_format": "channels_last"}, "inbound_nodes": [[["up4", 0, 0, {}]]], "shared_object_id": 26, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 71}}
?
regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "concat_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concat_3", "trainable": true, "dtype": "float32", "axis": 3}, "inbound_nodes": [[["drop3", 0, 0, {}], ["zero_padding2d_1", 0, 0, {}]]], "shared_object_id": 27, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 20, 17, 128]}, {"class_name": "TensorShape", "items": [null, 20, 17, 128]}]}
?

?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "conv3_up_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv3_up_0", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concat_3", 0, 0, {}]]], "shared_object_id": 29, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 72}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 17, 256]}}
?

?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "conv3_up", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv3_up", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv3_up_0", 0, 0, {}]]], "shared_object_id": 31, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 73}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 17, 128]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "up3_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling2D", "config": {"name": "up3_0", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "inbound_nodes": [[["conv3_up", 0, 0, {}]]], "shared_object_id": 32, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 74}}
?

?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "up3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "up3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["up3_0", 0, 0, {}]]], "shared_object_id": 34, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 75}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 34, 128]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "zero_padding2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_2", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 0]}, {"class_name": "__tuple__", "items": [0, 1]}]}, "data_format": "channels_last"}, "inbound_nodes": [[["up3", 0, 0, {}]]], "shared_object_id": 35, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 76}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "concat_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concat_2", "trainable": true, "dtype": "float32", "axis": 3}, "inbound_nodes": [[["conv2", 0, 0, {}], ["zero_padding2d_2", 0, 0, {}]]], "shared_object_id": 36, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 40, 35, 64]}, {"class_name": "TensorShape", "items": [null, 40, 35, 64]}]}
?

?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "conv2_up_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2_up_0", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 37}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concat_2", 0, 0, {}]]], "shared_object_id": 38, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 77}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 35, 128]}}
?

?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "conv2_up", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2_up", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 39}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2_up_0", 0, 0, {}]]], "shared_object_id": 40, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 78}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 35, 64]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "up2_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling2D", "config": {"name": "up2_0", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "inbound_nodes": [[["conv2_up", 0, 0, {}]]], "shared_object_id": 41, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 79}}
?

?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "up2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "up2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["up2_0", 0, 0, {}]]], "shared_object_id": 43, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 80}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80, 70, 64]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "concat_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concat_1", "trainable": true, "dtype": "float32", "axis": 3}, "inbound_nodes": [[["conv1", 0, 0, {}], ["up2", 0, 0, {}]]], "shared_object_id": 44, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 80, 70, 32]}, {"class_name": "TensorShape", "items": [null, 80, 70, 32]}]}
?

?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "conv1_up_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv1_up_0", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 45}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concat_1", 0, 0, {}]]], "shared_object_id": 46, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 81}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80, 70, 64]}}
?

?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "conv1_up", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv1_up", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": null, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 47}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv1_up_0", 0, 0, {}]]], "shared_object_id": 48, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 82}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80, 70, 32]}}
?
?kernel
	?bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv1_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv1_1", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 49}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 50}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv1_up", 0, 0, {}]]], "shared_object_id": 51, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 83}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80, 70, 32]}}
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "cropping2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Cropping2D", "config": {"name": "cropping2d", "trainable": true, "dtype": "float32", "cropping": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 1]}, {"class_name": "__tuple__", "items": [0, 1]}]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv1_1", 0, 0, {}]]], "shared_object_id": 52, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 84}}
?
	?iter

?decay
?learning_rate
?momentum
?rho
-rms?
.rms?
3rms?
4rms?
=rms?
>rms?
Crms?
Drms?
Mrms?
Nrms?
Srms?
Trms?
arms?
brms?
grms?
hrms?
urms?
vrms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms??rms?"
	optimizer
 "
trackable_list_wrapper
?
-0
.1
32
43
=4
>5
C6
D7
M8
N9
S10
T11
a12
b13
g14
h15
u16
v17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35"
trackable_list_wrapper
?
-0
.1
32
43
=4
>5
C6
D7
M8
N9
S10
T11
a12
b13
g14
h15
u16
v17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35"
trackable_list_wrapper
?
$regularization_losses
%trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
&	variables
?layer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
)regularization_losses
 ?layer_regularization_losses
*trainable_variables
?non_trainable_variables
?layers
?metrics
+	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:& 2conv1_0/kernel
: 2conv1_0/bias
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
?
/regularization_losses
 ?layer_regularization_losses
0trainable_variables
?non_trainable_variables
?layers
?metrics
1	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$  2conv1/kernel
: 2
conv1/bias
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
5regularization_losses
 ?layer_regularization_losses
6trainable_variables
?non_trainable_variables
?layers
?metrics
7	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
9regularization_losses
 ?layer_regularization_losses
:trainable_variables
?non_trainable_variables
?layers
?metrics
;	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:& @2conv2_0/kernel
:@2conv2_0/bias
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
@trainable_variables
?non_trainable_variables
?layers
?metrics
A	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv2/kernel
:@2
conv2/bias
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
?
Eregularization_losses
 ?layer_regularization_losses
Ftrainable_variables
?non_trainable_variables
?layers
?metrics
G	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Iregularization_losses
 ?layer_regularization_losses
Jtrainable_variables
?non_trainable_variables
?layers
?metrics
K	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'@?2conv3_0/kernel
:?2conv3_0/bias
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
?
Oregularization_losses
 ?layer_regularization_losses
Ptrainable_variables
?non_trainable_variables
?layers
?metrics
Q	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&??2conv3/kernel
:?2
conv3/bias
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
?
Uregularization_losses
 ?layer_regularization_losses
Vtrainable_variables
?non_trainable_variables
?layers
?metrics
W	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Yregularization_losses
 ?layer_regularization_losses
Ztrainable_variables
?non_trainable_variables
?layers
?metrics
[	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
]regularization_losses
 ?layer_regularization_losses
^trainable_variables
?non_trainable_variables
?layers
?metrics
_	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(??2conv4_0/kernel
:?2conv4_0/bias
 "
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
?
cregularization_losses
 ?layer_regularization_losses
dtrainable_variables
?non_trainable_variables
?layers
?metrics
e	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&??2conv4/kernel
:?2
conv4/bias
 "
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
?
iregularization_losses
 ?layer_regularization_losses
jtrainable_variables
?non_trainable_variables
?layers
?metrics
k	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
mregularization_losses
 ?layer_regularization_losses
ntrainable_variables
?non_trainable_variables
?layers
?metrics
o	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
qregularization_losses
 ?layer_regularization_losses
rtrainable_variables
?non_trainable_variables
?layers
?metrics
s	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$??2
up4/kernel
:?2up4/bias
 "
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
?
wregularization_losses
 ?layer_regularization_losses
xtrainable_variables
?non_trainable_variables
?layers
?metrics
y	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
{regularization_losses
 ?layer_regularization_losses
|trainable_variables
?non_trainable_variables
?layers
?metrics
}	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+??2conv3_up_0/kernel
:?2conv3_up_0/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)??2conv3_up/kernel
:?2conv3_up/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#?@2
up3/kernel
:@2up3/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*?@2conv2_up_0/kernel
:@2conv2_up_0/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2_up/kernel
:@2conv2_up/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"@ 2
up2/kernel
: 2up2/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)@ 2conv1_up_0/kernel
: 2conv1_up_0/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'  2conv1_up/kernel
: 2conv1_up/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:& 2conv1_1/kernel
:2conv1_1/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
 ?layer_regularization_losses
?trainable_variables
?non_trainable_variables
?layers
?metrics
?	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 85}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}, "shared_object_id": 55}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "root_mse", "dtype": "float32", "config": {"name": "root_mse", "dtype": "float32", "fn": "root_mse"}, "shared_object_id": 56}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
2:0 2RMSprop/conv1_0/kernel/rms
$:" 2RMSprop/conv1_0/bias/rms
0:.  2RMSprop/conv1/kernel/rms
":  2RMSprop/conv1/bias/rms
2:0 @2RMSprop/conv2_0/kernel/rms
$:"@2RMSprop/conv2_0/bias/rms
0:.@@2RMSprop/conv2/kernel/rms
": @2RMSprop/conv2/bias/rms
3:1@?2RMSprop/conv3_0/kernel/rms
%:#?2RMSprop/conv3_0/bias/rms
2:0??2RMSprop/conv3/kernel/rms
#:!?2RMSprop/conv3/bias/rms
4:2??2RMSprop/conv4_0/kernel/rms
%:#?2RMSprop/conv4_0/bias/rms
2:0??2RMSprop/conv4/kernel/rms
#:!?2RMSprop/conv4/bias/rms
0:.??2RMSprop/up4/kernel/rms
!:?2RMSprop/up4/bias/rms
7:5??2RMSprop/conv3_up_0/kernel/rms
(:&?2RMSprop/conv3_up_0/bias/rms
5:3??2RMSprop/conv3_up/kernel/rms
&:$?2RMSprop/conv3_up/bias/rms
/:-?@2RMSprop/up3/kernel/rms
 :@2RMSprop/up3/bias/rms
6:4?@2RMSprop/conv2_up_0/kernel/rms
':%@2RMSprop/conv2_up_0/bias/rms
3:1@@2RMSprop/conv2_up/kernel/rms
%:#@2RMSprop/conv2_up/bias/rms
.:,@ 2RMSprop/up2/kernel/rms
 : 2RMSprop/up2/bias/rms
5:3@ 2RMSprop/conv1_up_0/kernel/rms
':% 2RMSprop/conv1_up_0/bias/rms
3:1  2RMSprop/conv1_up/kernel/rms
%:# 2RMSprop/conv1_up/bias/rms
2:0 2RMSprop/conv1_1/kernel/rms
$:"2RMSprop/conv1_1/bias/rms
?2?
&__inference_model_layer_call_fn_203564
&__inference_model_layer_call_fn_204531
&__inference_model_layer_call_fn_204608
&__inference_model_layer_call_fn_204151?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_model_layer_call_and_return_conditional_losses_204770
A__inference_model_layer_call_and_return_conditional_losses_204946
A__inference_model_layer_call_and_return_conditional_losses_204260
A__inference_model_layer_call_and_return_conditional_losses_204369?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_202978?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????OE
?2?
/__inference_zero_padding2d_layer_call_fn_202991?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
J__inference_zero_padding2d_layer_call_and_return_conditional_losses_202985?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
(__inference_conv1_0_layer_call_fn_204955?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv1_0_layer_call_and_return_conditional_losses_204966?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_conv1_layer_call_fn_204975?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv1_layer_call_and_return_conditional_losses_204986?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_pool1_layer_call_fn_203003?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
A__inference_pool1_layer_call_and_return_conditional_losses_202997?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
(__inference_conv2_0_layer_call_fn_204995?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2_0_layer_call_and_return_conditional_losses_205006?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_conv2_layer_call_fn_205015?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv2_layer_call_and_return_conditional_losses_205026?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_pool2_layer_call_fn_203015?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
A__inference_pool2_layer_call_and_return_conditional_losses_203009?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
(__inference_conv3_0_layer_call_fn_205035?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv3_0_layer_call_and_return_conditional_losses_205046?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_conv3_layer_call_fn_205055?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv3_layer_call_and_return_conditional_losses_205066?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_drop3_layer_call_fn_205071
&__inference_drop3_layer_call_fn_205076?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_drop3_layer_call_and_return_conditional_losses_205081
A__inference_drop3_layer_call_and_return_conditional_losses_205093?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_pool3_layer_call_fn_203027?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
A__inference_pool3_layer_call_and_return_conditional_losses_203021?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
(__inference_conv4_0_layer_call_fn_205102?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv4_0_layer_call_and_return_conditional_losses_205113?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_conv4_layer_call_fn_205122?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv4_layer_call_and_return_conditional_losses_205133?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_drop4_layer_call_fn_205138
&__inference_drop4_layer_call_fn_205143?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_drop4_layer_call_and_return_conditional_losses_205148
A__inference_drop4_layer_call_and_return_conditional_losses_205160?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_up4_0_layer_call_fn_203046?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
A__inference_up4_0_layer_call_and_return_conditional_losses_203040?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
$__inference_up4_layer_call_fn_205169?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_up4_layer_call_and_return_conditional_losses_205180?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_zero_padding2d_1_layer_call_fn_203059?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
L__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_203053?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
)__inference_concat_3_layer_call_fn_205186?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_concat_3_layer_call_and_return_conditional_losses_205193?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv3_up_0_layer_call_fn_205202?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv3_up_0_layer_call_and_return_conditional_losses_205213?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv3_up_layer_call_fn_205222?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv3_up_layer_call_and_return_conditional_losses_205233?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_up3_0_layer_call_fn_203078?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
A__inference_up3_0_layer_call_and_return_conditional_losses_203072?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
$__inference_up3_layer_call_fn_205242?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_up3_layer_call_and_return_conditional_losses_205253?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_zero_padding2d_2_layer_call_fn_203091?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
L__inference_zero_padding2d_2_layer_call_and_return_conditional_losses_203085?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
)__inference_concat_2_layer_call_fn_205259?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_concat_2_layer_call_and_return_conditional_losses_205266?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv2_up_0_layer_call_fn_205275?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2_up_0_layer_call_and_return_conditional_losses_205286?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2_up_layer_call_fn_205295?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2_up_layer_call_and_return_conditional_losses_205306?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_up2_0_layer_call_fn_203110?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
A__inference_up2_0_layer_call_and_return_conditional_losses_203104?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
$__inference_up2_layer_call_fn_205315?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_up2_layer_call_and_return_conditional_losses_205326?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_concat_1_layer_call_fn_205332?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_concat_1_layer_call_and_return_conditional_losses_205339?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv1_up_0_layer_call_fn_205348?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv1_up_0_layer_call_and_return_conditional_losses_205359?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv1_up_layer_call_fn_205368?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv1_up_layer_call_and_return_conditional_losses_205379?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv1_1_layer_call_fn_205388?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv1_1_layer_call_and_return_conditional_losses_205398?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_cropping2d_layer_call_fn_203125?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
F__inference_cropping2d_layer_call_and_return_conditional_losses_203119?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?B?
$__inference_signature_wrapper_204454input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_202978?6-.34=>CDMNSTabghuv??????????????????8?5
.?+
)?&
input_1?????????OE
? "??<
:

cropping2d,?)

cropping2d?????????OE?
D__inference_concat_1_layer_call_and_return_conditional_losses_205339?|?y
r?o
m?j
*?'
inputs/0?????????PF 
<?9
inputs/1+??????????????????????????? 
? "-?*
#? 
0?????????PF@
? ?
)__inference_concat_1_layer_call_fn_205332?|?y
r?o
m?j
*?'
inputs/0?????????PF 
<?9
inputs/1+??????????????????????????? 
? " ??????????PF@?
D__inference_concat_2_layer_call_and_return_conditional_losses_205266?|?y
r?o
m?j
*?'
inputs/0?????????(#@
<?9
inputs/1+???????????????????????????@
? ".?+
$?!
0?????????(#?
? ?
)__inference_concat_2_layer_call_fn_205259?|?y
r?o
m?j
*?'
inputs/0?????????(#@
<?9
inputs/1+???????????????????????????@
? "!??????????(#??
D__inference_concat_3_layer_call_and_return_conditional_losses_205193?~?{
t?q
o?l
+?(
inputs/0??????????
=?:
inputs/1,????????????????????????????
? ".?+
$?!
0??????????
? ?
)__inference_concat_3_layer_call_fn_205186?~?{
t?q
o?l
+?(
inputs/0??????????
=?:
inputs/1,????????????????????????????
? "!????????????
C__inference_conv1_0_layer_call_and_return_conditional_losses_204966l-.7?4
-?*
(?%
inputs?????????PF
? "-?*
#? 
0?????????PF 
? ?
(__inference_conv1_0_layer_call_fn_204955_-.7?4
-?*
(?%
inputs?????????PF
? " ??????????PF ?
C__inference_conv1_1_layer_call_and_return_conditional_losses_205398n??7?4
-?*
(?%
inputs?????????PF 
? "-?*
#? 
0?????????PF
? ?
(__inference_conv1_1_layer_call_fn_205388a??7?4
-?*
(?%
inputs?????????PF 
? " ??????????PF?
A__inference_conv1_layer_call_and_return_conditional_losses_204986l347?4
-?*
(?%
inputs?????????PF 
? "-?*
#? 
0?????????PF 
? ?
&__inference_conv1_layer_call_fn_204975_347?4
-?*
(?%
inputs?????????PF 
? " ??????????PF ?
F__inference_conv1_up_0_layer_call_and_return_conditional_losses_205359n??7?4
-?*
(?%
inputs?????????PF@
? "-?*
#? 
0?????????PF 
? ?
+__inference_conv1_up_0_layer_call_fn_205348a??7?4
-?*
(?%
inputs?????????PF@
? " ??????????PF ?
D__inference_conv1_up_layer_call_and_return_conditional_losses_205379n??7?4
-?*
(?%
inputs?????????PF 
? "-?*
#? 
0?????????PF 
? ?
)__inference_conv1_up_layer_call_fn_205368a??7?4
-?*
(?%
inputs?????????PF 
? " ??????????PF ?
C__inference_conv2_0_layer_call_and_return_conditional_losses_205006l=>7?4
-?*
(?%
inputs?????????(# 
? "-?*
#? 
0?????????(#@
? ?
(__inference_conv2_0_layer_call_fn_204995_=>7?4
-?*
(?%
inputs?????????(# 
? " ??????????(#@?
A__inference_conv2_layer_call_and_return_conditional_losses_205026lCD7?4
-?*
(?%
inputs?????????(#@
? "-?*
#? 
0?????????(#@
? ?
&__inference_conv2_layer_call_fn_205015_CD7?4
-?*
(?%
inputs?????????(#@
? " ??????????(#@?
F__inference_conv2_up_0_layer_call_and_return_conditional_losses_205286o??8?5
.?+
)?&
inputs?????????(#?
? "-?*
#? 
0?????????(#@
? ?
+__inference_conv2_up_0_layer_call_fn_205275b??8?5
.?+
)?&
inputs?????????(#?
? " ??????????(#@?
D__inference_conv2_up_layer_call_and_return_conditional_losses_205306n??7?4
-?*
(?%
inputs?????????(#@
? "-?*
#? 
0?????????(#@
? ?
)__inference_conv2_up_layer_call_fn_205295a??7?4
-?*
(?%
inputs?????????(#@
? " ??????????(#@?
C__inference_conv3_0_layer_call_and_return_conditional_losses_205046mMN7?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0??????????
? ?
(__inference_conv3_0_layer_call_fn_205035`MN7?4
-?*
(?%
inputs?????????@
? "!????????????
A__inference_conv3_layer_call_and_return_conditional_losses_205066nST8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
&__inference_conv3_layer_call_fn_205055aST8?5
.?+
)?&
inputs??????????
? "!????????????
F__inference_conv3_up_0_layer_call_and_return_conditional_losses_205213p??8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
+__inference_conv3_up_0_layer_call_fn_205202c??8?5
.?+
)?&
inputs??????????
? "!????????????
D__inference_conv3_up_layer_call_and_return_conditional_losses_205233p??8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
)__inference_conv3_up_layer_call_fn_205222c??8?5
.?+
)?&
inputs??????????
? "!????????????
C__inference_conv4_0_layer_call_and_return_conditional_losses_205113nab8?5
.?+
)?&
inputs?????????
?
? ".?+
$?!
0?????????
?
? ?
(__inference_conv4_0_layer_call_fn_205102aab8?5
.?+
)?&
inputs?????????
?
? "!??????????
??
A__inference_conv4_layer_call_and_return_conditional_losses_205133ngh8?5
.?+
)?&
inputs?????????
?
? ".?+
$?!
0?????????
?
? ?
&__inference_conv4_layer_call_fn_205122agh8?5
.?+
)?&
inputs?????????
?
? "!??????????
??
F__inference_cropping2d_layer_call_and_return_conditional_losses_203119?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
+__inference_cropping2d_layer_call_fn_203125?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
A__inference_drop3_layer_call_and_return_conditional_losses_205081n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
A__inference_drop3_layer_call_and_return_conditional_losses_205093n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
&__inference_drop3_layer_call_fn_205071a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
&__inference_drop3_layer_call_fn_205076a<?9
2?/
)?&
inputs??????????
p
? "!????????????
A__inference_drop4_layer_call_and_return_conditional_losses_205148n<?9
2?/
)?&
inputs?????????
?
p 
? ".?+
$?!
0?????????
?
? ?
A__inference_drop4_layer_call_and_return_conditional_losses_205160n<?9
2?/
)?&
inputs?????????
?
p
? ".?+
$?!
0?????????
?
? ?
&__inference_drop4_layer_call_fn_205138a<?9
2?/
)?&
inputs?????????
?
p 
? "!??????????
??
&__inference_drop4_layer_call_fn_205143a<?9
2?/
)?&
inputs?????????
?
p
? "!??????????
??
A__inference_model_layer_call_and_return_conditional_losses_204260?6-.34=>CDMNSTabghuv??????????????????@?=
6?3
)?&
input_1?????????OE
p 

 
? "-?*
#? 
0?????????OE
? ?
A__inference_model_layer_call_and_return_conditional_losses_204369?6-.34=>CDMNSTabghuv??????????????????@?=
6?3
)?&
input_1?????????OE
p

 
? "-?*
#? 
0?????????OE
? ?
A__inference_model_layer_call_and_return_conditional_losses_204770?6-.34=>CDMNSTabghuv????????????????????<
5?2
(?%
inputs?????????OE
p 

 
? "-?*
#? 
0?????????OE
? ?
A__inference_model_layer_call_and_return_conditional_losses_204946?6-.34=>CDMNSTabghuv????????????????????<
5?2
(?%
inputs?????????OE
p

 
? "-?*
#? 
0?????????OE
? ?
&__inference_model_layer_call_fn_203564?6-.34=>CDMNSTabghuv??????????????????@?=
6?3
)?&
input_1?????????OE
p 

 
? " ??????????OE?
&__inference_model_layer_call_fn_204151?6-.34=>CDMNSTabghuv??????????????????@?=
6?3
)?&
input_1?????????OE
p

 
? " ??????????OE?
&__inference_model_layer_call_fn_204531?6-.34=>CDMNSTabghuv????????????????????<
5?2
(?%
inputs?????????OE
p 

 
? " ??????????OE?
&__inference_model_layer_call_fn_204608?6-.34=>CDMNSTabghuv????????????????????<
5?2
(?%
inputs?????????OE
p

 
? " ??????????OE?
A__inference_pool1_layer_call_and_return_conditional_losses_202997?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
&__inference_pool1_layer_call_fn_203003?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
A__inference_pool2_layer_call_and_return_conditional_losses_203009?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
&__inference_pool2_layer_call_fn_203015?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
A__inference_pool3_layer_call_and_return_conditional_losses_203021?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
&__inference_pool3_layer_call_fn_203027?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
$__inference_signature_wrapper_204454?6-.34=>CDMNSTabghuv??????????????????C?@
? 
9?6
4
input_1)?&
input_1?????????OE"??<
:

cropping2d,?)

cropping2d?????????OE?
A__inference_up2_0_layer_call_and_return_conditional_losses_203104?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
&__inference_up2_0_layer_call_fn_203110?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
?__inference_up2_layer_call_and_return_conditional_losses_205326???I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
$__inference_up2_layer_call_fn_205315???I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
A__inference_up3_0_layer_call_and_return_conditional_losses_203072?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
&__inference_up3_0_layer_call_fn_203078?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
?__inference_up3_layer_call_and_return_conditional_losses_205253???J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
$__inference_up3_layer_call_fn_205242???J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????@?
A__inference_up4_0_layer_call_and_return_conditional_losses_203040?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
&__inference_up4_0_layer_call_fn_203046?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
?__inference_up4_layer_call_and_return_conditional_losses_205180?uvJ?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
$__inference_up4_layer_call_fn_205169?uvJ?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
L__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_203053?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_zero_padding2d_1_layer_call_fn_203059?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
L__inference_zero_padding2d_2_layer_call_and_return_conditional_losses_203085?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_zero_padding2d_2_layer_call_fn_203091?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_zero_padding2d_layer_call_and_return_conditional_losses_202985?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_zero_padding2d_layer_call_fn_202991?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????
scene(X) :- scene(X,_).
object(ID) :- obj(ID,_,_,_,_,_,_).
position(ID,X,Y) :- obj(ID,X,Y,_,_,_,_).
has_size(ID,SIZE) :- obj(ID,_,_,_,_,_,SIZE).
has_color(ID,COLOR) :- obj(ID,_,_,_,COLOR,_,_).
has_material(ID,MATERIAL):- obj(ID,_,_,MATERIAL,_,_,_).
has_shape(ID,SHAPE) :- obj(ID,_,_,_,_,SHAPE,_).
left_of(ID,ID') :- position(ID,X,Y), position(ID',X',Y'), state(T',ID'), ID!=ID', X<X'.
right_of(ID,ID') :- position(ID,X,Y), position(ID',X',Y'), state(T',ID'), ID!=ID', X>=X'.
in_front_of(ID,ID') :- position(ID,X,Y), position(ID',X',Y'), state(T',ID'), ID!=ID', Y>Y'.
behind_of(ID,ID') :- position(ID,X,Y), position(ID',X',Y'), state(T',ID'), ID!=ID', Y<=Y'.
state(T+1,ID) :- unique(T), state(T,ID).
:- unique(T), state(T,ID), state(T,ID'), ID!=ID'.
state(T+1,ID) :- relate_left(T), state(T,ID'), left_of(ID,ID').
state(T+1,ID) :- relate_right(T), state(T,ID'), right_of(ID,ID').
state(T+1,ID) :- relate_front(T), state(T,ID'), in_front_of(ID,ID') .
state(T+1,ID) :- relate_behind(T), state(T,ID'), behind_of(ID,ID').
int(T+1,V) :- count(T), #count{ ID : state(T,ID) } = V.
bool(T+1,yes) :- exist(T), state(T,ID).
bool(T+1,no) :- exist(T), not bool(T+1,yes).
state(T+1,ID) :- filter_large(T), state(T,ID), has_size(ID,large).
state(T+1,ID) :- filter_small(T), state(T,ID), has_size(ID,small).
state(T+1,ID) :- filter_gray(T), state(T,ID), has_color(ID,gray).
state(T+1,ID) :- filter_red(T), state(T,ID), has_color(ID,red).
state(T+1,ID) :- filter_blue(T), state(T,ID), has_color(ID,blue).
state(T+1,ID) :- filter_green(T), state(T,ID), has_color(ID,green).
state(T+1,ID) :- filter_brown(T), state(T,ID), has_color(ID,brown).
state(T+1,ID) :- filter_purple(T), state(T,ID), has_color(ID,purple).
state(T+1,ID) :- filter_cyan(T), state(T,ID), has_color(ID,cyan).
state(T+1,ID) :- filter_yellow(T), state(T,ID), has_color(ID,yellow).
state(T+1,ID) :- filter_metal(T), state(T,ID), has_material(ID,metal).
state(T+1,ID) :- filter_rubber(T), state(T,ID), has_material(ID,rubber).
state(T+1,ID) :- filter_sphere(T), state(T,ID), has_shape(ID,sphere).
state(T+1,ID) :- filter_cylinder(T), state(T,ID), has_shape(ID,cylinder).
state(T+1,ID) :- filter_cube(T), state(T,ID), has_shape(ID,cube).
size(T+1,SIZE) :- query_size(T), state(T,ID), has_size(ID,SIZE).
color(T+1,COLOR) :- query_color(T), state(T,ID), has_color(ID,COLOR).
material(T+1,MATERIAL) :- query_material(T), state(T,ID), has_material(ID,MATERIAL).
shape(T+1,SHAPE) :- query_shape(T), state(T,ID), has_shape(ID,SHAPE).
state(T+1,ID) :- and(T,T'), state(T,ID), state(T',ID).
state(T+1,ID) :- or(T,T'), state(T,ID).
state(T+1,ID') :- or(T,T'), state(T',ID').
bool(T+1, yes) :- boolean_negation(T), bool(T, no).
bool(T+1, no) :- boolean_negation(T), not bool(T+1, yes).
state(T+1,ID') :- same_size(T), state(T,ID), has_size(ID,SIZE), has_size(ID',SIZE), ID!=ID'.
state(T+1,ID') :- same_color(T), state(T,ID), has_color(ID,COLOR), has_color(ID',COLOR), ID!=ID'.
state(T+1,ID') :- same_material(T), state(T,ID), has_material(ID,MATERIAL), has_material(ID',MATERIAL), ID!=ID'.
state(T+1,ID') :- same_shape(T), state(T,ID), has_shape(ID,SHAPE), has_shape(ID',SHAPE), ID!=ID'.
bool(T+1,yes) :- equal_integer(T,T'), int(T,V), int(T',V'), V=V'.
bool(T+1,no) :- equal_integer(T,T'), not bool(T+1,yes).
bool(T+1,yes) :- less_than(T,T'), int(T,V), int(T',V'), V<V'.
bool(T+1,no) :- less_than(T,T'), not bool(T+1,yes).
bool(T+1,yes) :- greater_than(T,T'), int(T,V), int(T',V'), V>V'.
bool(T+1,no) :- greater_than(T,T'), not bool(T+1,yes).
bool(T+1,yes) :- equal_size(T,T'), size(T,V), size(T',V'), V=V'.
bool(T+1,no) :- equal_size(T,T'), not bool(T+1,yes).
bool(T+1,yes) :- equal_color(T,T'), color(T,V), color(T',V'), V=V'.
bool(T+1,no) :- equal_color(T,T'), not bool(T+1,yes).
bool(T+1,yes) :- equal_material(T,T'), material(T,V), material(T',V'), V=V'.
bool(T+1,no) :- equal_material(T,T'), not bool(T+1,yes).
bool(T+1,yes) :- equal_shape(T,T'), shape(T,V), shape(T',V'), V=V'.
bool(T+1,no) :- equal_shape(T,T'), not bool(T+1,yes).
state(0,ID) :- object(ID).
state(T+1,ID) :- scene(T), object(ID).
ans(V) :- end(T), size(T,V).
ans(V) :- end(T), color(T,V).
ans(V) :- end(T), material(T,V).
ans(V) :- end(T), shape(T,V).
ans(V) :- end(T), bool(T,V).
ans(V) :- end(T), int(T,V).
:- not ans(_).
#show ans/1.
% Added rules to handle new instances

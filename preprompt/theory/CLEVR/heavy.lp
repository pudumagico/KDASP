% Uniqueness rule/constraint
state(T+1,ID) :- unique(T), state(T,ID).
:- unique(T), state(T,ID), state(T,ID'), ID!=ID'.

% Spatial relation rules
state(T+1,ID) :- relate_left(T), state(T,ID'), left_of(ID,ID').
state(T+1,ID) :- relate_right(T), state(T,ID'), right_of(ID,ID').

% Count rule
int(T+1,V) :- count(T), #count{ ID : state(T,ID) } = V.

% Exist rule
bool(T+1,yes) :- exist(T), state(T,ID).
bool(T+1,no) :- exist(T), not bool(T+1,yes).

% Filtering rules
state(T+1,ID) :- filter_large(T), state(T,ID), has_size(ID,large).
state(T+1,ID) :- filter_small(T), state(T,ID), has_size(ID,small).
state(T+1,ID) :- filter_gray(T), state(T,ID), has_color(ID,gray).
state(T+1,ID) :- filter_red(T), state(T,ID), has_color(ID,red).
state(T+1,ID) :- filter_blue(T), state(T,ID), has_color(ID,blue).
state(T+1,ID) :- filter_green(T), state(T,ID), has_color(ID,green).
state(T+1,ID) :- filter_brown(T), state(T,ID), has_color(ID,brown).

% Query functions
size(T+1,SIZE) :- query_size(T), state(T,ID), has_size(ID,SIZE).
color(T+1,COLOR) :- query_color(T), state(T,ID), has_color(ID,COLOR).

% Logical operators
state(T+1,ID) :- and(T,T'), state(T,ID), state(T',ID).

state(T+1,ID) :- or(T,T'), state(T,ID).
state(T+1,ID') :- or(T,T'), state(T',ID').

bool(T+1, yes) :- boolean_negation(T), bool(T, no).
bool(T+1, no) :- boolean_negation(T), not bool(T+1, yes).

% Same-attribute relations
state(T+1,ID') :- same_size(T), state(T,ID), has_size(ID,SIZE), has_size(ID',SIZE), ID!=ID'.
state(T+1,ID') :- same_color(T), state(T,ID), has_color(ID,COLOR), has_color(ID',COLOR), ID!=ID'.

% % Integer comparison
bool(T+1,yes) :- equal_integer(T,T'), int(T,V), int(T',V'), V=V'.
bool(T+1,no) :- equal_integer(T,T'), not bool(T+1,yes).

bool(T+1,yes) :- less_than(T,T'), int(T,V), int(T',V'), V<V'.
bool(T+1,no) :- less_than(T,T'), not bool(T+1,yes).


% Attribute comparison
bool(T+1,yes) :- equal_size(T,T'), size(T,V), size(T',V'), V=V'.
bool(T+1,no) :- equal_size(T,T'), not bool(T+1,yes).

bool(T+1,yes) :- equal_color(T,T'), color(T,V), color(T',V'), V=V'.
bool(T+1,no) :- equal_color(T,T'), not bool(T+1,yes).


% State rules
state(0,ID) :- object(ID).
state(T+1,ID) :- scene(T), object(ID).

scene(X) :- scene(X,Y).

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

% Derive answer (T must equal the last point in time)
ans(V) :- end(T), size(T,V).
ans(V) :- end(T), color(T,V).
ans(V) :- end(T), material(T,V).
ans(V) :- end(T), shape(T,V).
ans(V) :- end(T), bool(T,V).
ans(V) :- end(T), int(T,V).

:- not ans(_).

#show ans/1.

% Added rules to handle new instances

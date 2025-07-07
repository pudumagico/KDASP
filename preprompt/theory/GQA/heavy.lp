% Original Theory
is_attr_value(ID, X) :- query(TO, TI, ATTR), state(TI, ID), has_attr(ID, ATTR, X).
is_attr(X) :- query(TO, TI, ATTR), state(TI, ID), has_attr(ID, ATTR, X).

state(TO,ID) :- scene(TO), object(ID).

state(TO,ID) :- select(TO, TI, CLASS), state(TI, ID), has_attr(ID, class, CLASS).
state(TO,ID) :- filter(TO, TI, ATTR, VALUE), state(TI, ID), has_attr(ID, ATTR, VALUE).
{state(TO,ID): state(TI,ID)} = 1 :- unique(TO, TI).

state(TO,ID) :- select(TO, TI, CLASS), state(TI, ID), has_attr(ID, class, CLASS).
state(TO,ID) :- filter(TO, TI, ATTR, VALUE), state(TI, ID), has_attr(ID, ATTR, VALUE).
state(TO,ID) :- filter_any(TO, TI, VALUE), state(TI, ID), has_attr(ID, ATTR, VALUE).
state(TO, ID') :- relate(TO, TI, CLASS, REL, subject), state(TI, ID), has_attr(ID', class, CLASS), has_rel(ID', REL, ID).
state(TO, ID') :- relate(TO, TI, CLASS, REL, object), state(TI, ID), has_attr(ID', class, CLASS), has_rel(ID, REL, ID').
state(TO, ID') :- relate_any(TO, TI, REL, subject), state(TI, ID), has_rel(ID', REL, ID).
state(TO, ID') :- relate_any(TO, TI, REL, object), state(TI, ID), has_rel(ID, REL, ID').
state(TO, ID') :- relate_attr(TO, TI, CLASS, ATTR), state(TI, ID), has_attr(ID, ATTR, VALUE), has_attr(ID', class, CLASS), has_attr(ID', ATTR, VALUE'), VALUE==VALUE', ID!=ID'.

bool(TO,yes) :- and(TO, TI0, TI1), bool(TI0,yes), bool(TI1,yes).
bool(TO,no) :- and(TO, TI0, TI1), not bool(TO,yes).
bool(TO,yes) :- or(TO, TI0, TI1), bool(TI0,yes).
bool(TO,yes) :- or(TO, TI0, TI1), bool(TI1,yes).
bool(TO,no) :- or(TO, TI0, TI1), not bool(TO,yes).
{state(TO,ID): state(TI,ID)} = 1 :- unique(TO, TI).
:~ unique(TO, TI), state(TO,ID), has_obj_weight(ID, P). [P, (TO, ID)]
state(TO, ID) :- negate(TO, TI0, TI1), state(TI1, ID), not state(TI0, ID).

ans(V) :- end(TO), attr_value(TO,V).
ans(V) :- end(TO), attr(TO,V).
ans(V) :- end(TO), rel(TO,V).
ans(V) :- end(TO), bool(TO,V).

% Added rules to handle new instances

% End of Theory

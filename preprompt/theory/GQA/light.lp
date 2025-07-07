% Original Theory
is_attr_value(ID, X) :- query(TO, TI, ATTR), state(TI, ID), has_attr(ID, ATTR, X).
is_attr(X) :- query(TO, TI, ATTR), state(TI, ID), has_attr(ID, ATTR, X).

state(TO,ID) :- scene(TO), object(ID).

state(TO,ID) :- select(TO, TI, CLASS), state(TI, ID), has_attr(ID, class, CLASS).
state(TO,ID) :- filter(TO, TI, ATTR, VALUE), state(TI, ID), has_attr(ID, ATTR, VALUE).
{state(TO,ID): state(TI,ID)} = 1 :- unique(TO, TI).

ans(V) :- end(TO), attr_value(TO,V).
ans(V) :- end(TO), attr(TO,V).
ans(V) :- end(TO), rel(TO,V).
ans(V) :- end(TO), bool(TO,V).

% Added rules to handle new instances

% End of Theory

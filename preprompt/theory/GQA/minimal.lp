% Original Theory
state(TO,ID) :- scene(TO), object(ID).

ans(V) :- end(TO), attr_value(TO,V).
ans(V) :- end(TO), attr(TO,V).
ans(V) :- end(TO), rel(TO,V).
ans(V) :- end(TO), bool(TO,V).

% Added rules to handle new instances

% End of Theory

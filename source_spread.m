function [L] = source_spread(GW, id, sigma)
%gaussian source amplitude spread
D = shortest_paths(GW, id);
L= normpdf(D, 0, sigma);


end

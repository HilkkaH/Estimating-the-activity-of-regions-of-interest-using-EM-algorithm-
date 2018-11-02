function [GW] = get_dist_matrix(surf)

Nvc = size(surf.p, 1);
Ntri = size(surf.e, 1);

GW = spalloc(Nvc, Nvc, 3*Ntri);
for ii = 1:Ntri
    tri = surf.e(ii, :);
    GW(tri(1), tri(2)) = norm(surf.p(tri(1), :)-surf.p(tri(2), :));
    GW(tri(2), tri(3)) = norm(surf.p(tri(2), :)-surf.p(tri(3), :));
    GW(tri(3), tri(1)) = norm(surf.p(tri(3), :)-surf.p(tri(1), :)); 
end
end

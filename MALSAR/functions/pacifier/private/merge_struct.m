function s = merge_struct( s1 ,s2 )
% merge_struct : Merge two structures
% 
%   $Revision: 0.1.0 $  $Date: 2012/6/15 $
% 
  s = s1;
  
  names = fieldnames( s2 );
  for k = 1:length( names )
    if isfield( s, names{k} )
      if isstruct( s.(names{k}) )
        s.(names{k}) = merge_struct( s.(names{k}), s2.(names{k}) );
      else
        s.(names{k}) = s2.(names{k});
      end
    else
      s.(names{k}) = s2.(names{k});
    end
  end
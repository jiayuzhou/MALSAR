function options = pnopt_optimset( varargin )
% pnopt_optimset : Set options for PNOPT
%
% options = pnopt_optimset( 'param1', val1, 'param2', val2,... ) creates an options
%   structure in which the named parameters have the specified values. Default
%   values are used for parameters that are not specified.
%   
% pnopt_optimset with no input arguments and no output arguments displays all 
%   parameter names and their default values.
%
%   $Revision: 0.8.0 $  $Date: 2012/12/01 $
% 
  % Print out possible values of options.
  if nargin == 0 && nargout == 0
    fprintf('Default parameters for PNOPT:\n');
    fprintf('     curv_param : [  positive scalar |     0.9 ] \n');    
    fprintf('          debug : [          logical |       0 ] \n');
    fprintf('     desc_param : [  positive scalar |  0.0001 ] \n');
    fprintf('        display : [ positive integer |      10 ] \n');
    fprintf('      Lbfgs_mem : [ positive integer |      50 ] \n');
    fprintf('  backtrack_mem : [ positive integer |      10 ] \n');
    fprintf('       maxfunEv : [ positive integer |    5000 ] \n');
    fprintf('        maxIter : [ positive integer |     500 ] \n');
    fprintf('         method : [           string |   Lbfgs ] \n');
    fprintf(' subprob_solver : [           string |  sparsa ] \n');
    fprintf('           ftol : [  positive scalar |    1e-9 ] \n');
    fprintf('      optim_tol : [  positive scalar |    1e-6 ] \n');
    fprintf('           xtol : [  positive scalar |    1e-9 ] \n');
    fprintf('\n');
    return;
  end

  Names = [
    'curv_param     '
    'debug          '
    'desc_param     '
    'display        '
    'Lbfgs_mem      '
    'backtrack_mem  '
    'maxfunEv       '
    'maxIter        '
    'method         '
    'sparsa_options '
    'subprob_solver '
    'Tfocs_opts     '
    'ftol           '
    'optim_tol      '
    'xtol           '
    'useMex         '
    ];
  
  [m,n] = size(Names); %#ok<NASGU>
  names = lower(Names);

  % Combine all leading options structures o1, o2, ... in l1Set(o1,o2,...).
  options = [];
  for j = 1:m
    eval(['options.' Names(j,:) '= [];']);
  end
  i = 1;
  while i <= nargin
    arg = varargin{i};
    if ischar(arg), break; end
    if ~isempty(arg)                      % [] is a valid options argument
      if ~isa(arg,'struct')
        error(sprintf(['Expected argument %d to be a string parameter name ' ...
          'or an options structure\ncreated with SetPNoptOptions.'], i)); %#ok<SPERR>
      end
      for j = 1:m
        if any(strcmp(fieldnames(arg),deblank(Names(j,:))))
          eval(['val = arg.' Names(j,:) ';']);
        else
          val = [];
        end
        if ~isempty(val)
          eval(['options.' Names(j,:) '= val;']);
        end
      end
    end
    i = i + 1;
  end
  
  % A finite state machine to parse name-value pairs.
  if rem(nargin-i+1,2) ~= 0
    error('Arguments must occur in name-value pairs.');
  end
  expectval = 0;                          % start expecting a name, not a value
  while i <= nargin
    arg = varargin{i};

    if ~expectval
      if ~ischar(arg)
         error('Expected argument %d to be a string parameter name.', i);
      end

      lowArg = lower(arg);
      j = strmatch(lowArg,names);
      if isempty(j)                       % if no matches
        error('Unrecognized parameter name ''%s''.', arg);
      elseif length(j) > 1                % if more than one match
        % Check for any exact matches (in case any names are subsets of others)
        k = strmatch(lowArg,names,'exact');
        if length(k) == 1
          j = k;
        else
          msg = sprintf('Ambiguous parameter name ''%s'' ', arg);
          msg = [msg '(' deblank(Names(j(1),:))]; %#ok<AGROW>
          for k = j(2:length(j))'
            msg = [msg ', ' deblank(Names(k,:))]; %#ok<AGROW>
          end
          error('%s).', msg);
        end
      end
      expectval = 1;                      % we expect a value next

    else
      eval(['options.' Names(j,:) '= arg;']);
      expectval = 0;
      
    end
    i = i + 1;
  end

  if expectval
    error('Expected value for parameter ''%s''.', arg);
  end


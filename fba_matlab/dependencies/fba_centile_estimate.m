function centile=fba_centile_estimate(age_var,fba_var,subid,adjust,offset_param,age_centiles,fba_centiles,centiles_tested)
% fba_centile_estimate computes the centile value based on actual age and functional brain age estimate
% and utilises precomputed centiles spanning the data in the combined FBA
% model (D1&D2)

%inputs
%age_var = actual age
%fba_var = functional brain age
%subid = subject id
%adjust = uses choose whether fba is adjusted to the linear fit; 
%offset_param = regression offset for the combined FBA model
%age_centiles = age centiles; computed via GAMLSS in Rstudio
%fba_centiles = fba centiles; computed via GAMLSS in Rstudio. 
%This is a m x n matrix where m is the number of centiles tested (see
%below) and n is the number of subjects in the combined FBA model
%centiles_tested = centile binning set to [0.5:0.5:99.5] 

%output
%centile = centile value based on nearest FBA value in fba_centiles

% linear offset for projecting onto final chart
if strcmp(adjust,'offset')
    fba_var = fba_var+(age_var-((offset_param(2)*age_var)));
elseif strcmp(adjust,'none')
    %no adjustment of fba
end

[~,age_nearest_idx] = min(bsxfun(@(x,y)abs(x-y),age_var(subid),age_centiles'),[],2);
testcen=fba_centiles(:,age_nearest_idx);

[~,fba_nearest_idx] = min(bsxfun(@(x,y)abs(x-y),fba_var(subid),testcen'),[],2);
centile=(centiles_tested(fba_nearest_idx));

% Test figure, uncomment to see centile fit
figure, plot(age_var,fba_var,'ko')
hold on, plot(age_var(subid),fba_var(subid),'go','MarkerFaceColor','g','MarkerSize',10)
hold on, plot(age_centiles,fba_centiles(fba_nearest_idx,:),'b','LineWidth',2)


end
function [p1, p2] = tf_correlation(dat, fs1)

fs = 64;
dat1 = resample(dat', fs, fs1)';
A = size(dat1);
N = 512;
block_no = floor(length(dat1)/N);
x = [0:fs:N-fs/2] + fs/2;
y = [0:fs/8:N/2-fs/16] + fs/16;
for z0 = 1:A(1)
    [Pxx, f] = pwelch(dat(z0,:), hamming(2^11), 2^10, 2^12, fs);
    fr = 2.^[0.6:0.1:4]-1; P = zeros(1,length(fr)-1);
    for ii = 1:length(fr)-1
       rx = find(f>=fr(ii) & f<(fr(ii+1)));
       P(ii) = mean(Pxx(rx)); 
    end  
    outliers = 1; x = 1:length(P); logf = log2(fr); logP = log2(P); idx = zeros(1,length(P));
    while outliers ~= 0
        x1 = logf(idx==0); y1 = logP(idx==0);
        B = regress(y1', [x1' ones(size(x1'))]);
        res = y1 - (B(1)*x1+B(2)*ones(1,length(x1)));
        [~, idx] = rmoutliers(res);
        outliers = sum(idx);
    end
    sc = 2^(B(2)).*(y./16).^B(1);
    tfr = [];
    for z1 = 1:block_no
        r1 = (z1-1)*N+1; r2 = z1*N;
        tfrep = wvd(dat1(z0, r1:r2), fs,'smoothedPseudo', hamming(N/4-1), hamming(N/32-1), 'NumFrequencyPoints', N,'NumTimePoints', N);
        % resample tfrep
        tfdum = conv2(tfrep(1:N/2,:), ones(fs/8,fs), 'same');
        tfr = [tfr ; tfdum(y,x)'];
    end
    tf{z0} = tfr;
end
p1 = zeros(A(1)); p2 = p1; 
for z0 = 1:A(1)
    for z1 = z0+1:A(1)
        p1(z0, z1) = corr(tf{z0}(:), tf{z1}(:));
        tfx1 = tf{z0}.*(1./sc); tfx2 = tf{z1}.*(1./sc);
        p2(z0, z1) = corr(tfx1(:), tfx2(:));
    end
end

% visualizations
% figure;
% subplot(1,2,1); 
% imagesc(abs(p2))
% set(gca, 'Ytick', [1:19], 'Yticklabel', str1)
% set(gca, 'Xtick', [1:19], 'Xticklabel', str1)
% set(gca, 'Position', [0.05 0.1 0.5 0.85], 'fontsize', 14)
% pp = abs(p2); pp = round(pp*63)+1;
% c = colormap; 
% val = [1 2 ; 1 4 ; 2 1 ; 2 2 ; 2 3 ; 2 4 ; 2 5 ; 3 1 ; 3 2 ; 3 3 ; 3 4 ; 3 5 ; 4 1 ; 4 2 ; 4 3 ; 4 4 ; 4 5 ; 5 2 ; 5 4]
% val(:,1) = 6-val(:,1);
% x = linspace(-2.5, 2.5, 500); r = 2.5;
% subplot(1,2,2)
% hold on; %plot(val(:,2), val(:,1),'o')
% plot(x+3, real(sqrt(r^2-x.^2))+3, 'b.'); plot(x+3, -real(sqrt(r^2-x.^2))+3, 'b.'); 
% for ii = 1:19
%     for jj = ii+1:19
%         plot([val(ii,2) val(jj,2)], [val(ii,1) val(jj,1)], 'linewidth',2 , 'color', c(pp(ii,jj),:))
%     end
% end
% for ii = 1:19;
%    text(val(ii,2), val(ii,1), str1{ii}, 'HorizontalAlignment', 'center', 'BackgroundColor', [1 1 1], 'fontsize', 16, 'fontweight', 'bold') 
% end
% set(gca, 'Position', [0.575 0.05 0.4 0.9], 'fontsize', 14)
% axis off
% 
% 
% 
% 
% 
% 
% 
    
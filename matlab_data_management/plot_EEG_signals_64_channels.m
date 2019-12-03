
[hdr, record] = edfread("S001R01.edf");

%%

figure;
CM = winter(10);

count = 1;
for i=1:6:36
   subplot(6,1,count);
   x = mod(i,7)+1;
   bla = record(i,1:8000);
   minVal = min(bla);
   maxVal = max(bla);
   norm_data = (bla - minVal) / ( maxVal - minVal );
   plot(norm_data,'LineWidth',2,'Color',CM(count,:))
   xlim([1 8000])
   axis off
   count = count+1;
end
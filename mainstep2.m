clear;
close all;

n_average=10;
n_fold=5;


%%%cor_score_matrixp? feature-feature correlation matrix;
%%%cor_label_indexp: feature-label correlation vector;


path = './dataset'
dirs=dir(path)
% dircell=struct2cell(dirs)'
% filenames=dircell(:,1) 
num = size(dirs,1);
% ini_data = (np.load('./dataset/' + dir + '/correlation/cor_label_indexp.dat'))
% feature_map_whole_one_line=zeros(1,3)
%   csvwrite(strcat('.\dataset\', dirs(3).name, '\feature_map\filter4\feature_map_one_line.dat'),feature_map_whole_one_line);
% % 
 filter_size=4;
for oood=3:num
    load(strcat('.\dataset\', dirs(oood).name, '\correlation\cor_label_indexp.dat'));
    load(strcat('.\dataset\', dirs(oood).name, '\correlation\cor_score_matrixp.dat'));
    load(strcat('.\dataset\', dirs(oood).name, '\correlation\num_feature.dat'));
    
  
%     num_feature=24;

   
    feature_map_whole=zeros(n_average,n_fold,num_feature,num_feature)
    index_order_matrix=1
    index_order_label=1
    for num_average=1:n_average
        for num_fold=1:n_fold
             cor_score_matrix=zeros(num_feature,num_feature);
            cor_label_index=zeros(1,num_feature)

            for  mm=1:num_feature
                for  hh=1:num_feature
                  cor_score_matrix(mm,hh)=cor_score_matrixp(index_order_matrix)
                  index_order_matrix=index_order_matrix+1


                end

            end

            for kkkk=1:num_feature
               cor_label_index(1,kkkk)=int32(cor_label_indexp(index_order_label,1 )+1);
               index_order_label=index_order_label+1
            end

                [n_feature1,n_feature_2]=size(cor_score_matrix)
                feature_map=zeros(n_feature1,n_feature1)
                perm_2_row=randperm(n_feature1)

                feature_map(1,:)=cor_label_index
                total_need_comput=n_feature1*(n_feature1-1);
                i=0;

                for i=1:n_feature1-1                                                                                                                                                                           


                for jj=1:n_feature1-1
                    store_index=zeros(1,2);
                     cor_score_matrix1=cor_score_matrix;
                       ini_index_feature=[1:n_feature1];
                     if(jj==1)
                         cor_score_matrix1_next=cor_score_matrix1;
                         ini_index_feature_next=ini_index_feature;
                     end
                   new_feature_length=length(ini_index_feature_next)
                       x=binvar(new_feature_length,1);
                        a=ones(1,new_feature_length);
                        f=x'*cor_score_matrix1_next*x;
                        constrain=[];

                    options=sdpsettings('solver','bmibnb') ; 
                    if jj==1
                        b_set_1=[feature_map(i,jj:jj+1)];            
                        b1=zeros(length(b_set_1),new_feature_length)
                         for tt=1:length(b_set_1)
                             b1(tt,find(ini_index_feature_next==b_set_1(tt)))=1;
                         end
                        [uni_r uni_c]=size(unique(b_set_1))

                            diagnostics = optimize([a*x==min(uni_c+2,new_feature_length),b1*x==1],-f,options);

                    else
                        constrain=[];
                            b_set_1=[feature_map(i,jj:jj+1),feature_map(i+1,jj)];             
                            b1=zeros(length(b_set_1),new_feature_length)
                         for tt=1:length(b_set_1)
                             b1(tt,find(ini_index_feature_next==b_set_1(tt)))=1;
                         end
                         [uni_r uni_c]=size(unique(b_set_1))


                                 diagnostics = optimize([a*x==min(uni_c+1,new_feature_length),b1*x==1],-f,options);

                    end
                    decision=value(x) 
                    y_value=value(f)
                    [row,col]=find(decision==1)
                    pp=0       
                        if(jj==1)
                            for kk=1:length(row)
                               row(kk)=ini_index_feature_next(row(kk));
                              if(~ismember(row(kk),[feature_map(i,jj:jj+1)]))
                              pp=pp+1
                              store_index(1,pp)=row(kk);               
                              end          
                            end
                            feature_map(i+1,jj)=store_index(1,1);
                            feature_map(i+1,jj+1)=store_index(1,2);
                            new_element=2;
                %                  end
                             delete_row=[];
                            for kkk=1:new_element
                                 if(~ismember(store_index(1,kkk),[feature_map(i,jj+1:jj+2),feature_map(i+1,jj+1)])) 
                                    delete_row=[delete_row,store_index(1,kkk)];   

                                 end

                            end
                                for hh=1:jj
                               if(~ismember(feature_map(i+1,hh),[feature_map(i,jj+1:jj+2),feature_map(i+1,jj+1)])) 
                                  delete_row=[delete_row,feature_map(i+1,hh)]
                              end
                              end

                                 cor_score_matrix1(delete_row,:)=[];
                              cor_score_matrix1(:,delete_row)=[];
                              ini_index_feature(delete_row)=[];
                %                   end

                              cor_score_matrix1_next=cor_score_matrix1;
                              ini_index_feature_next =ini_index_feature;


                        else
                               for kk=1:min(uni_c+1,new_feature_length)
                                    row(kk)=ini_index_feature_next(row(kk));
                                   if(~ismember(row(kk),[feature_map(i,jj:jj+1),feature_map(i+1,jj)]))
                                   pp=pp+1
                                   store_index(1,pp)=row(kk);               
                                    end          
                               end
                               feature_yuan=[1:n_feature1];
                               if store_index(1,1)==0
                                   for gg=1:n_feature1
                                       if(~ismember(feature_yuan(gg),feature_map(i+1,1:jj+1)))
                                       store_index(1,1)=feature_yuan(gg);
                                       break;
                                       end
                                   end
                               end
                               feature_map(i+1,jj+1)=store_index(1,1)
                               delete_row=[];
                               for kkk=1
                                 if(jj<n_feature1-1)
                                  if(~ismember(store_index(1,1),[feature_map(i,jj+1:jj+2),feature_map(i+1,jj+1)])) 
                                     delete_row=[delete_row,store_index(1,kkk)]
                                  end
                                 else
                                      delete_row=[delete_row,store_index(1,kkk)]
                                 end


                               end 
                                if(jj<n_feature1-1)
                              for hh=1:jj
                               if(~ismember(feature_map(i+1,hh),[feature_map(i,jj+1:jj+2),feature_map(i+1,jj+1)])) 
                                  delete_row=[delete_row,feature_map(i+1,hh)]
                              end
                              end
                              cor_score_matrix1(delete_row,:)=[];
                              cor_score_matrix1(:,delete_row)=[];
                              ini_index_feature(delete_row)=[];

                              cor_score_matrix1_next=cor_score_matrix1;
                              ini_index_feature_next =ini_index_feature;
                        end
                        end


                end

                end
                for  uu=1:num_feature      
           feature_map_whole(num_average,num_fold,uu,:)=feature_map(uu,:)
          end
        end
    end
  

    feature_map_whole_one_line=zeros(n_average*n_fold*num_feature*num_feature,1)
    index=1;
    for f=1:n_average
        for s=1:n_fold
            for t=1:num_feature
                for u=1:num_feature
                    feature_map_whole_one_line(index)=feature_map_whole(f,s,t,u);
                    index=index+1;

                end
            end


        end
    end
     csvwrite(strcat('.\dataset-test\', dirs(oood).name, '\feature_map\filter4\feature_map_one_line.dat'),feature_map_whole_one_line);
end

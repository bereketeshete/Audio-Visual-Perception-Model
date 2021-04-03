
function data = load_data(data_dir)
    data_files = dir(data_dir);
    data_files = data_files(3:end);
    ndata = size(data_files,1);
    data = cell(ndata,1);
    
    for idx = 1:ndata
        temp=load(fullfile(data_dir, data_files(idx,1).name));
        data{idx,1} = temp.data;
    end      
end
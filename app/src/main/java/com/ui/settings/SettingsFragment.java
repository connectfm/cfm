package com.ui.settings;


import android.content.Context;
import android.os.Build;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.fragment.app.Fragment;

import com.example.cfm.R;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;

public class SettingsFragment extends Fragment {


    private EditText bias;
    private EditText radius;
    private Context context;


    // first load existing val. check if edit text is changed
    @RequiresApi(api = Build.VERSION_CODES.R)
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {
        //protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //setContentView(R.layout.fragment_settings);
        View root = inflater.inflate(R.layout.fragment_settings, container, false);


        Integer bias_val;
        Integer radius_val;
        bias = (EditText) root.findViewById(R.id.set_bias_field);
        radius = (EditText) root.findViewById(R.id.set_radius_field);


        context = getActivity().getApplicationContext();
        String bias_filename = "settings_bias.txt";
        String radius_filename = "settings_radius.txt";

        String bias_string = read_file(context, bias_filename);
        String radius_string = read_file(context, bias_filename);
        // if no existing settings
        if (bias_string.length() < 1)  // if no existing settings, set to 50
            bias_string = update_settings(50, bias_filename); // updates internal storage and the edittext

        if (radius_string.toString().length() < 1)
            radius_string = update_settings(50, radius_filename);

        bias.setText(bias_string);
        radius.setText(radius_string);

        bias.addTextChangedListener(new TextWatcher() {
            public void afterTextChanged(Editable s) {
                int new_val;
                if (s.length() > 0)
                    new_val = Integer.parseInt(s.toString());
                else
                    new_val = 50;
                update_settings(new_val, bias_filename); // don't need to update text because it's automatically updated as an edit text
            }

            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
            }

            public void onTextChanged(CharSequence s, int start, int before,
                                      int count) {

            }
        });

        radius.addTextChangedListener(new TextWatcher() {
            public void afterTextChanged(Editable s) {
                int new_val;
                if (s.length() > 0)
                    new_val = Integer.parseInt(s.toString());
                else
                    new_val = 50;
                update_settings(new_val, radius_filename); // don't need to update text because it's automatically updated as an edit text
            }

            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
            }

            public void onTextChanged(CharSequence s, int start, int before,
                                      int count) {

            }
        });
        return root;
    }




    private String update_settings(int val, String filename){
        String sval = Integer.toString(val);
        writeFileOnInternalStorage(context, filename, sval);
        return sval;
    }


    public void writeFileOnInternalStorage(Context mcoContext, String sFileName, String sBody){
        File dir = new File(mcoContext.getFilesDir(), "mydir");
        if(!dir.exists()){
            dir.mkdir();
        }

        try {
            File gpxfile = new File(dir, sFileName);
            FileWriter writer = new FileWriter(gpxfile);
            writer.append(sBody);
            writer.flush();
            writer.close();
        } catch (Exception e){
            e.printStackTrace();
        }
    }

    public String read_file(Context context, String filename) {
        try {
            FileInputStream fis = context.openFileInput(filename);
            InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
            BufferedReader bufferedReader = new BufferedReader(isr);
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                sb.append(line).append("\n");
            }
            return sb.toString();
        } catch (FileNotFoundException e) {
            return "";
        } catch (UnsupportedEncodingException e) {
            return "";
        } catch (IOException e) {
            return "";
        }
    }


}
